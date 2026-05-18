import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import pickle
import random

from douzero.env.game import GameEnv

EnvCard2RealCard = {
    3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'
}

_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']
_POS_NAME  = {
    'landlord':        '地主',
    'landlord_down':   '下家',
    'landlord_across': '对面',
    'landlord_up':     '上家',
}


def cards2str(cards):
    return ' '.join(EnvCard2RealCard[c] for c in sorted(cards)) if cards else 'PASS'


def load_agent(position, path):
    """Load RandomAgent or DeepAgent depending on path."""
    if path == 'random':
        from douzero.evaluation.random_agent import RandomAgent
        return RandomAgent()
    else:
        from douzero.evaluation.deep_agent import DeepAgent
        return DeepAgent(position, path)


class DeterministicAgent:
    """Holds a pre-set action so env.step() uses the same action we already printed."""
    def __init__(self):
        self.action = []

    def act(self, infoset):
        return self.action


def simulate(game_data, agents, seed=None):
    if seed is not None:
        random.seed(seed)

    print('=== 发牌情况 ===')
    print(f'地主(33张): {cards2str(game_data["landlord"])}')
    print(f'底牌( 8张): {cards2str(game_data["eight_landlord_cards"])}')
    print(f'下家(25张): {cards2str(game_data["landlord_down"])}')
    print(f'对面(25张): {cards2str(game_data["landlord_across"])}')
    print(f'上家(25张): {cards2str(game_data["landlord_up"])}')
    print()

    # DeterministicAgent proxies are what GameEnv sees;
    # the real agents decide the action externally.
    proxies = {p: DeterministicAgent() for p in _POSITIONS}
    env = GameEnv(proxies)
    env.card_play_init(game_data)

    print('=== 游戏过程 ===')
    step = 0
    while not env.game_over:
        pos    = env.acting_player_position
        action = agents[pos].act(env.game_infoset)   # real agent decides
        proxies[pos].action = action                  # proxy forwards it

        hand_cards  = list(env.info_sets[pos].player_hand_cards)
        hand_before = len(hand_cards)
        hand_after  = hand_before - len(action)
        tmp = hand_cards[:]
        for c in action:
            tmp.remove(c)
        remaining = tmp

        step += 1
        name = _POS_NAME[pos]

        if action:
            print(f'第{step:3d}步  {name}  出牌: {cards2str(action)}')
            print(f'         剩余{hand_after:2d}张: {cards2str(remaining) if remaining else "（空）"}')
        else:
            print(f'第{step:3d}步  {name}  PASS')
            print(f'         剩余{hand_before:2d}张: {cards2str(hand_cards)}')

        env.step()

    print()
    print('=== 游戏结束 ===')
    winner    = env.get_winner()
    bomb_num  = env.get_bomb_num()
    landlord_score = 3 if winner == 'landlord' else -3
    farmer_score   = -1 if winner == 'landlord' else 1
    print(f'获胜方: {"地主" if winner == "landlord" else "农民"}')
    print(f'炸弹数: {bomb_num}  倍数: x1（固定）')
    print(f'得分  — 地主: {landlord_score:+d}  农民: {farmer_score:+d}')


def get_parser():
    parser = argparse.ArgumentParser(description='DouZero 单局模拟（4人双副牌）')
    parser.add_argument('--data',            default='eval_data.pkl', type=str)
    parser.add_argument('--game_index',      default=0,    type=int)
    parser.add_argument('--seed',            default=None, type=int)
    parser.add_argument('--landlord',        default='random', type=str,
                        help='地主模型路径，或 random')
    parser.add_argument('--landlord_down',   default='random', type=str,
                        help='下家模型路径，或 random')
    parser.add_argument('--landlord_across', default='random', type=str,
                        help='对面模型路径，或 random')
    parser.add_argument('--landlord_up',     default='random', type=str,
                        help='上家模型路径，或 random')
    return parser


if __name__ == '__main__':
    flags = get_parser().parse_args()

    with open(flags.data, 'rb') as f:
        data = pickle.load(f)

    total = len(data)
    idx   = flags.game_index
    if idx < 0 or idx >= total:
        print(f'game_index 超出范围，数据集共 {total} 局（0 ~ {total - 1}）')
        raise SystemExit(1)

    model_paths = {
        'landlord':        flags.landlord,
        'landlord_down':   flags.landlord_down,
        'landlord_across': flags.landlord_across,
        'landlord_up':     flags.landlord_up,
    }
    print(f'模拟第 {idx} 局（共 {total} 局）')
    for p, path in model_paths.items():
        print(f'  {_POS_NAME[p]}: {path}')
    print()

    agents = {p: load_agent(p, path) for p, path in model_paths.items()}
    simulate(data[idx], agents, seed=flags.seed)
