import argparse
import pickle
import numpy as np

# 2-deck, 108 cards
deck = []
for i in range(3, 15):
    deck.extend([i] * 8)
deck.extend([17] * 8)
deck.extend([20, 20, 30, 30])


def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator (4-player)')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=10000, type=int)
    return parser


def generate():
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {
        'landlord':             sorted(_deck[:25] + _deck[100:108]),
        'landlord_down':        sorted(_deck[25:50]),
        'landlord_across':      sorted(_deck[50:75]),
        'landlord_up':          sorted(_deck[75:100]),
        'eight_landlord_cards': sorted(_deck[100:108]),
    }
    return card_play_data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'
    print('output_pickle:', output_pickle)
    print('generating data...')
    data = [generate() for _ in range(flags.num_games)]
    print('saving pickle file...')
    with open(output_pickle, 'wb') as g:
        pickle.dump(data, g, pickle.HIGHEST_PROTOCOL)
