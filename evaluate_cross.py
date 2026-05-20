"""
Cross win rate evaluation for 4-player DouDizhu.

Each deal is played twice on the SAME card distribution:
  Batch 1 — AI as landlord, opponents at the 3 farmer positions.
  Batch 2 — opponent as landlord, AI at the 3 farmer positions.

This cancels out card-distribution luck. Position-specific checkpoints are
loaded for each seat.

Total games = len(eval_data) * 2.
Cross win rate = AI wins / total games.

Usage:
    python evaluate_cross.py --fps 217600
    python evaluate_cross.py --fps 217600 --opponent rule
    python evaluate_cross.py --fps 217600 --eval_data eval_data_1000.pkl --gpu_device 7
"""

import argparse
import multiprocessing as mp
import os
import pickle
from copy import deepcopy

from douzero.env.game import GameEnv
from douzero.evaluation.simulation import (
    data_allocation_per_worker,
    load_card_play_models,
    mp_simulate,
)


def _run_batch(card_play_data_list, card_play_model_path_dict, num_workers):
    if num_workers <= 1:
        # Sequential fallback (avoids Windows multiprocessing PermissionError)
        total_l_wins = total_f_wins = total_l_scores = total_f_scores = 0
        for card_play_data in card_play_data_list:
            players = load_card_play_models(card_play_model_path_dict)
            env = GameEnv(players)
            env.card_play_init(deepcopy(card_play_data))
            while not env.game_over:
                env.step()
            total_l_wins   += env.num_wins['landlord']
            total_f_wins   += env.num_wins['farmer']
            total_l_scores += env.num_scores['landlord']
            total_f_scores += env.num_scores['farmer']
        return total_l_wins, total_f_wins, total_l_scores, total_f_scores

    per_worker = data_allocation_per_worker(card_play_data_list, num_workers)

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for data_chunk in per_worker:
        p = ctx.Process(target=mp_simulate,
                        args=(data_chunk, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    num_landlord_wins = num_farmer_wins = 0
    num_landlord_scores = num_farmer_scores = 0

    for _ in range(num_workers):
        result = q.get()
        num_landlord_wins   += result[0]
        num_farmer_wins     += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores   += result[3]

    return num_landlord_wins, num_farmer_wins, num_landlord_scores, num_farmer_scores


def _play_one(card_play_data, card_play_model_path_dict):
    """Play a single game sequentially, return winner ('landlord' or 'farmer')."""
    players = load_card_play_models(card_play_model_path_dict)
    env = GameEnv(players)
    env.card_play_init(deepcopy(card_play_data))
    while not env.game_over:
        env.step()
    return env.get_winner()


def _print_game(card_play_data, idx):
    """Print one deal's card distribution."""
    EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                        9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K',
                        14: 'A', 17: '2', 20: 'X', 30: 'D'}

    def c2s(cards):
        return ' '.join(EnvCard2RealCard[c] for c in sorted(cards))

    print('-' * 60)
    print(f'  Game #{idx} — 地主无论谁当都能赢（牌运局）')
    print('-' * 60)
    print(f'  地主:     {c2s(card_play_data["landlord"])}  ({len(card_play_data["landlord"])}张)')
    print(f'  底牌:     {c2s(card_play_data["eight_landlord_cards"])}')
    print(f'  下家:     {c2s(card_play_data["landlord_down"])}  (25张)')
    print(f'  对面:     {c2s(card_play_data["landlord_across"])}  (25张)')
    print(f'  上家:     {c2s(card_play_data["landlord_up"])}  (25张)')
    print('-' * 60)


def evaluate_cross(fps, opponent='random', eval_data='eval_data.pkl', num_workers=5):
    """Cross win rate evaluation.

    Batch 1: AI at landlord (landlord_weights_{fps}.ckpt),
             opponents at farmer seats.
    Batch 2: opponent at landlord,
             AI at farmer seats (landlord_down/across/up_weights_{fps}.ckpt).

    Same deals are used for both batches.  Total games = len(eval_data) * 2.
    """
    ai_landlord_model = f'./douzero_checkpoints/douzero/landlord_weights_{fps}.ckpt'
    ai_farmer_models = {
        'landlord_down':   f'./douzero_checkpoints/douzero/landlord_down_weights_{fps}.ckpt',
        'landlord_across': f'./douzero_checkpoints/douzero/landlord_across_weights_{fps}.ckpt',
        'landlord_up':     f'./douzero_checkpoints/douzero/landlord_up_weights_{fps}.ckpt',
    }

    with open(eval_data, 'rb') as f:
        data = pickle.load(f)

    # Model configs for the two batches
    batch1_dict = {
        'landlord':        ai_landlord_model,
        'landlord_down':   opponent,
        'landlord_across': opponent,
        'landlord_up':     opponent,
    }
    batch2_dict = {
        'landlord':        opponent,
        'landlord_down':   ai_farmer_models['landlord_down'],
        'landlord_across': ai_farmer_models['landlord_across'],
        'landlord_up':     ai_farmer_models['landlord_up'],
    }

    # --- Quick sequential scan: find the first deal where landlord always wins ---
    print('Scanning for a deal where landlord wins regardless of who plays it...')
    found = False
    for i, deal in enumerate(data):
        try:
            w1 = _play_one(deal, batch1_dict)
        except Exception as e:
            print(f'  [game {i}] batch1 error: {type(e).__name__}: {e}')
            continue
        try:
            w2 = _play_one(deal, batch2_dict)
        except Exception as e:
            print(f'  [game {i}] batch2 error: {type(e).__name__}: {e}')
            continue
        if w1 == 'landlord' and w2 == 'landlord':
            _print_game(deal, i)
            found = True
            break
    if not found:
        print('(none found — ok)')
    print()

    # --- Full multiprocess evaluation ---
    print('Running full evaluation...')

    l_wins, f_wins, l_scores, f_scores = _run_batch(data, batch1_dict, num_workers)
    ai_wins_as_landlord = l_wins
    ai_scores_as_landlord = l_scores
    games_as_landlord = l_wins + f_wins

    l_wins, f_wins, l_scores, f_scores = _run_batch(data, batch2_dict, num_workers)
    ai_wins_as_farmer = f_wins
    ai_scores_as_farmer = f_scores
    games_as_farmer = l_wins + f_wins

    total_games = games_as_landlord + games_as_farmer
    total_ai_wins = ai_wins_as_landlord + ai_wins_as_farmer
    total_ai_scores = ai_scores_as_landlord + ai_scores_as_farmer

    cross_wp = total_ai_wins / total_games if total_games > 0 else 0
    cross_adp = total_ai_scores / total_games if total_games > 0 else 0

    print('=' * 60)
    print('Cross Win Rate Evaluation')
    print('=' * 60)
    print(f'  fps:               {fps}')
    print(f'  opponent:          {opponent}')
    print(f'  eval_data:         {eval_data}')
    print(f'  unique deals:      {games_as_landlord}')
    print(f'  total games:       {total_games}')
    print(f'  num_workers:       {num_workers}')
    print('-' * 60)
    print(f'  [Batch 1] AI landlord  vs  opponent farmers ({opponent})')
    print(f'    landlord         {ai_landlord_model}')
    print(f'  [Batch 2] opponent landlord ({opponent})  vs  AI farmers')
    for pos, m in ai_farmer_models.items():
        print(f'    {pos:<18} {m}')
    print('-' * 60)
    if games_as_landlord > 0:
        print(f'  AI as landlord:     {ai_wins_as_landlord}/{games_as_landlord} wins'
              f'  (WP={ai_wins_as_landlord / games_as_landlord:.4f})')
    if games_as_farmer > 0:
        print(f'  AI as farmer team:  {ai_wins_as_farmer}/{games_as_farmer} wins'
              f'  (WP={ai_wins_as_farmer / games_as_farmer:.4f})')
    print('-' * 60)
    print(f'  Cross WP:           {cross_wp:.4f}  ({total_ai_wins}/{total_games})')
    print(f'  Cross ADP:          {cross_adp:.4f}')
    print('=' * 60)

    return {
        'cross_wp': cross_wp,
        'cross_adp': cross_adp,
        'total_games': total_games,
        'total_ai_wins': total_ai_wins,
        'ai_wins_as_landlord': ai_wins_as_landlord,
        'games_as_landlord': games_as_landlord,
        'wp_as_landlord': ai_wins_as_landlord / games_as_landlord if games_as_landlord > 0 else 0,
        'ai_wins_as_farmer': ai_wins_as_farmer,
        'games_as_farmer': games_as_farmer,
        'wp_as_farmer': ai_wins_as_farmer / games_as_farmer if games_as_farmer > 0 else 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DouDizhu 4-player Cross Win Rate Evaluation')
    parser.add_argument('--fps', type=int, required=True,
                        help='Training frames of the checkpoint, e.g. 217600')
    parser.add_argument('--opponent', type=str, default='random',
                        choices=['random', 'rule'],
                        help='Opponent type (default: random)')
    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device

    evaluate_cross(
        fps=args.fps,
        opponent=args.opponent,
        eval_data=args.eval_data,
        num_workers=args.num_workers,
    )
