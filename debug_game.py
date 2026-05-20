"""Debug: play the same deal twice — once AI as landlord, once AI as farmer."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import os
import pickle

from douzero.env.game import GameEnv

EnvCard2RealCard = {
    3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'
}

_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']
_POS_NAME = {
    'landlord': '地主',
    'landlord_down': '下家',
    'landlord_across': '对面',
    'landlord_up': '上家',
}


def cards2str(cards):
    return ' '.join(EnvCard2RealCard[c] for c in sorted(cards)) if cards else 'PASS'


def load_agent(position, path):
    if path == 'random':
        from douzero.evaluation.random_agent import RandomAgent
        return RandomAgent()
    elif path == 'rule':
        from douzero.evaluation.rule_agent import RuleAgent
        return RuleAgent(position)
    else:
        from douzero.evaluation.deep_agent import DeepAgent
        return DeepAgent(position, path)


class DeterministicAgent:
    def __init__(self):
        self.action = []

    def act(self, infoset):
        return self.action


def play_one(game_data, agents, title):
    proxies = {p: DeterministicAgent() for p in _POSITIONS}
    env = GameEnv(proxies)
    env.card_play_init(game_data)

    print(f'=== {title} ===')
    print(f'地主(33张): {cards2str(game_data["landlord"])}')
    print(f'底牌( 8张): {cards2str(game_data["eight_landlord_cards"])}')
    print(f'下家(25张): {cards2str(game_data["landlord_down"])}')
    print(f'对面(25张): {cards2str(game_data["landlord_across"])}')
    print(f'上家(25张): {cards2str(game_data["landlord_up"])}')
    print()

    step = 0
    while not env.game_over:
        pos = env.acting_player_position
        action = agents[pos].act(env.game_infoset)
        proxies[pos].action = action

        hand_cards = list(env.info_sets[pos].player_hand_cards)
        hand_before = len(hand_cards)
        tmp = hand_cards[:]
        for c in action:
            tmp.remove(c)
        remaining = tmp

        step += 1
        name = _POS_NAME[pos]

        if action:
            print(f'第{step:3d}步  {name}  出牌: {cards2str(action)}')
            print(f'         剩余{len(remaining):2d}张: {cards2str(remaining) if remaining else "（空）"}')
        else:
            print(f'第{step:3d}步  {name}  PASS')
            print(f'         剩余{hand_before:2d}张: {cards2str(hand_cards)}')

        env.step()

    winner = env.get_winner()
    bomb_num = env.get_bomb_num()
    print()
    print(f'>> 结果: {"地主胜" if winner == "landlord" else "农民胜"}  炸弹数: {bomb_num}')
    print()

    return winner


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Debug: same deal, two roles')
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--eval_data', type=str, default='eval_data_100.pkl')
    parser.add_argument('--opponent', type=str, default='random')
    parser.add_argument('--game_index', type=int, default=0)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device

    ai_landlord_model = f'./douzero_checkpoints/douzero/landlord_weights_{args.fps}.ckpt'
    ai_farmer_models = {
        'landlord_down':   f'./douzero_checkpoints/douzero/landlord_down_weights_{args.fps}.ckpt',
        'landlord_across': f'./douzero_checkpoints/douzero/landlord_across_weights_{args.fps}.ckpt',
        'landlord_up':     f'./douzero_checkpoints/douzero/landlord_up_weights_{args.fps}.ckpt',
    }

    with open(args.eval_data, 'rb') as f:
        data = pickle.load(f)

    idx = args.game_index
    game = data[idx]

    # Round 1: AI as landlord
    agents1 = {
        'landlord':        load_agent('landlord', ai_landlord_model),
        'landlord_down':   load_agent('landlord_down', args.opponent),
        'landlord_across': load_agent('landlord_across', args.opponent),
        'landlord_up':     load_agent('landlord_up', args.opponent),
    }
    w1 = play_one(game, agents1, f'Round 1: AI=地主  Opp={args.opponent}  (game #{idx})')

    # Round 2: AI as farmer
    agents2 = {
        'landlord':        load_agent('landlord', args.opponent),
        'landlord_down':   load_agent('landlord_down', ai_farmer_models['landlord_down']),
        'landlord_across': load_agent('landlord_across', ai_farmer_models['landlord_across']),
        'landlord_up':     load_agent('landlord_up', ai_farmer_models['landlord_up']),
    }
    w2 = play_one(game, agents2, f'Round 2: AI=农民  Opp={args.opponent}  (game #{idx})')

    print('=== 总结 ===')
    print(f'  Round 1 (AI地主): {"AI胜" if w1 == "landlord" else "AI负"}')
    print(f'  Round 2 (AI农民): {"AI胜" if w2 == "farmer" else "AI负"}')
