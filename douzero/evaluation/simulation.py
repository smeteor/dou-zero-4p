import multiprocessing as mp
import pickle

from douzero.env.game import GameEnv

_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']


def load_card_play_models(card_play_model_path_dict):
    players = {}
    for position in _POSITIONS:
        path = card_play_model_path_dict[position]
        if path == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif path == 'rule':
            from .rule_agent import RuleAgent
            players[position] = RuleAgent(position)
        elif path == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, path)
    return players


def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    players = load_card_play_models(card_play_model_path_dict)
    env = GameEnv(players)
    for card_play_data in card_play_data_list:
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()
    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']))


def data_allocation_per_worker(card_play_data_list, num_workers):
    per_worker = [[] for _ in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        per_worker[idx % num_workers].append(data)
    return per_worker


def evaluate(landlord, landlord_down, landlord_across, landlord_up, eval_data, num_workers):
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    per_worker = data_allocation_per_worker(card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord':        landlord,
        'landlord_down':   landlord_down,
        'landlord_across': landlord_across,
        'landlord_up':     landlord_up,
    }

    num_landlord_wins = num_farmer_wins = 0
    num_landlord_scores = num_farmer_scores = 0

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

    for _ in range(num_workers):
        result = q.get()
        num_landlord_wins   += result[0]
        num_farmer_wins     += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores   += result[3]

    total = num_landlord_wins + num_farmer_wins
    print('WP results:')
    print(f'landlord : Farmers — {num_landlord_wins/total:.4f} : {num_farmer_wins/total:.4f}')
    print('ADP results:')
    print(f'landlord : Farmers — {num_landlord_scores/total:.4f} : {3*num_farmer_scores/total:.4f}')
