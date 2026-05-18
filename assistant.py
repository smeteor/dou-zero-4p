import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import collections

from douzero.env.move_generator import MovesGener
from douzero.env.move_detector import get_move_type
from douzero.env import move_selector as ms
from douzero.env.game import InfoSet, _POSITIONS, _NEXT_POS, bombs

# Card name <-> env value mappings
RealCard2EnvCard = {
    '3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,
    'J':11,'Q':12,'K':13,'A':14,'2':17,'X':20,'D':30,
    'j':11,'q':12,'k':13,'a':14,'x':20,'d':30,
}
EnvCard2RealCard = {
    3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',
    11:'J',12:'Q',13:'K',14:'A',17:'2',20:'X',30:'D',
}

_POS_NAME = {
    'landlord':'地主', 'landlord_down':'下家',
    'landlord_across':'对面', 'landlord_up':'上家',
}

# Full 2-deck (108 cards)
_FULL_DECK = []
for _v in list(range(3, 15)) + [17]:
    _FULL_DECK.extend([_v] * 8)
_FULL_DECK.extend([20, 20, 30, 30])


def parse_cards(s):
    """Parse a space-separated card string into sorted env-value list."""
    s = s.strip().upper()
    if not s or s in ('PASS', 'P', '过', '不要'):
        return []
    tokens = s.split()
    result = []
    for t in tokens:
        if t not in RealCard2EnvCard:
            raise ValueError(f'无法识别的牌: {t}')
        result.append(RealCard2EnvCard[t])
    return sorted(result)


def cards2str(cards):
    return ' '.join(EnvCard2RealCard[c] for c in sorted(cards)) if cards else 'PASS'


def remove_cards(hand, played):
    """Remove played cards from hand (in-place copy)."""
    h = list(hand)
    for c in played:
        h.remove(c)
    return h


class GameAssistant:
    def __init__(self, my_position, my_hand, eight_landlord_cards, model=None):
        self.my_pos = my_position
        self.model  = model

        # Per-position hand tracking (we only know our own exactly)
        self.hand = {p: [] for p in _POSITIONS}
        self.hand[my_position] = sorted(my_hand)

        # Initial card counts
        if my_position == 'landlord':
            self.initial_counts = {p: 25 for p in _POSITIONS}
            self.initial_counts['landlord'] = 33
        else:
            self.initial_counts = {p: 25 for p in _POSITIONS}
            self.initial_counts['landlord'] = 33

        self.eight_landlord_cards = sorted(eight_landlord_cards)
        self.played_cards   = {p: [] for p in _POSITIONS}
        self.action_seq     = []
        self.last_move_dict = {p: [] for p in _POSITIONS}
        self.bomb_num       = 0
        self.last_pid       = 'landlord'
        self.acting_pos     = 'landlord'

        # Unknown pool: full deck minus my hand minus bottom cards (landlord already has them)
        unknown = list(_FULL_DECK)
        for c in my_hand:
            unknown.remove(c)
        if my_position != 'landlord':
            for c in eight_landlord_cards:
                unknown.remove(c)
        self.unknown_pool = unknown  # cards held by the other 3 players

    # ------------------------------------------------------------------ #
    def _estimate_other_hand_cards(self):
        """Best-effort estimate: all unknown cards not yet played by others."""
        pool = list(self.unknown_pool)
        for p in _POSITIONS:
            if p == self.my_pos:
                continue
            for c in self.played_cards[p]:
                if c in pool:
                    pool.remove(c)
        return sorted(pool)

    def _num_cards_left(self):
        result = {}
        for p in _POSITIONS:
            if p == self.my_pos:
                result[p] = len(self.hand[p])
            else:
                result[p] = self.initial_counts[p] - len(self.played_cards[p])
        return result

    def _get_last_move(self):
        for move in reversed(self.action_seq):
            if move:
                return move
        return []

    def _get_legal_actions(self):
        pos   = self.acting_pos
        cards = self.hand[pos] if pos == self.my_pos else []
        mg    = MovesGener(cards)

        if self.last_pid == pos:
            rival_move = []
        else:
            rival_move = self._get_last_move()

        rival_info = get_move_type(rival_move)
        rival_type = rival_info['type']
        rival_len  = rival_info.get('len', 1)

        from douzero.env.move_detector import (
            TYPE_0_PASS, TYPE_1_SINGLE, TYPE_2_PAIR, TYPE_3_TRIPLE,
            TYPE_4_BOMB, TYPE_5_KING_BOMB, TYPE_7_3_2,
            TYPE_9_SERIAL_PAIR, TYPE_10_SERIAL_TRIPLE, TYPE_12_SERIAL_3_2,
            TYPE_16_PENTA_BOMB, TYPE_17_HEXA_BOMB,
            TYPE_18_HEPTA_BOMB, TYPE_19_OCTA_BOMB,
        )
        _bomb_types = {TYPE_4_BOMB, TYPE_5_KING_BOMB, TYPE_16_PENTA_BOMB,
                       TYPE_17_HEXA_BOMB, TYPE_18_HEPTA_BOMB, TYPE_19_OCTA_BOMB}

        moves = []
        if rival_type == TYPE_0_PASS:
            moves = mg.gen_moves()
        elif rival_type == TYPE_1_SINGLE:
            moves = ms.filter_type_1_single(mg.gen_type_1_single(), rival_move)
        elif rival_type == TYPE_2_PAIR:
            moves = ms.filter_type_2_pair(mg.gen_type_2_pair(), rival_move)
        elif rival_type == TYPE_3_TRIPLE:
            moves = ms.filter_type_3_triple(mg.gen_type_3_triple(), rival_move)
        elif rival_type == TYPE_7_3_2:
            moves = ms.filter_type_7_3_2(mg.gen_type_7_3_2(), rival_move)
        elif rival_type == TYPE_9_SERIAL_PAIR:
            moves = ms.filter_type_9_serial_pair(
                mg.gen_type_9_serial_pair(repeat_num=rival_len), rival_move)
        elif rival_type == TYPE_10_SERIAL_TRIPLE:
            moves = ms.filter_type_10_serial_triple(
                mg.gen_type_10_serial_triple(repeat_num=rival_len), rival_move)
        elif rival_type == TYPE_12_SERIAL_3_2:
            moves = ms.filter_type_12_serial_3_2(
                mg.gen_type_12_serial_3_2(repeat_num=rival_len), rival_move)
        elif rival_type in _bomb_types:
            moves = ms.filter_bombs(mg.gen_all_bombs(), rival_move)
        elif rival_type == TYPE_5_KING_BOMB:
            moves = []

        if rival_type not in _bomb_types:
            moves = moves + mg.gen_all_bombs()
        if rival_move:
            moves = moves + [[]]

        for m in moves:
            m.sort()
        return moves

    def _build_infoset(self):
        pos  = self.my_pos
        info = InfoSet(pos)
        info.player_hand_cards  = list(self.hand[pos])
        info.other_hand_cards   = self._estimate_other_hand_cards()
        info.eight_landlord_cards = list(self.eight_landlord_cards)
        info.card_play_action_seq = list(self.action_seq)
        info.played_cards       = {p: list(v) for p, v in self.played_cards.items()}
        info.last_move          = self._get_last_move()
        info.last_two_moves     = self.action_seq[-2:] if len(self.action_seq) >= 2 else self.action_seq[:]
        info.last_move_dict     = {p: list(v) for p, v in self.last_move_dict.items()}
        info.num_cards_left_dict = self._num_cards_left()
        info.bomb_num           = self.bomb_num
        info.last_pid           = self.last_pid
        info.legal_actions      = self._get_legal_actions()
        info.all_handcards      = {p: list(self.hand[p]) for p in _POSITIONS}
        return info

    def _ai_recommend(self):
        """Return AI-recommended action using model, or sorted legal actions."""
        infoset = self._build_infoset()
        legal   = infoset.legal_actions

        if self.model:
            import torch, numpy as np
            from douzero.env.env import get_obs
            obs     = get_obs(infoset)
            z_batch = torch.from_numpy(obs['z_batch']).float()
            x_batch = torch.from_numpy(obs['x_batch']).float()
            with torch.no_grad():
                vals = self.model.forward(z_batch, x_batch, return_value=True)['values']
            vals = vals.detach().cpu().numpy().flatten()
            best = legal[int(vals.argmax())]
            ranked = sorted(zip(vals, legal), reverse=True)
            return best, ranked
        else:
            return legal[0] if legal else [], [(0, m) for m in legal]

    def apply_action(self, pos, action):
        """Apply a move for any position and advance state."""
        if action:
            self.last_pid = pos
        if action in bombs:
            self.bomb_num += 1
        self.last_move_dict[pos] = list(action)
        self.action_seq.append(list(action))
        self.played_cards[pos].extend(action)
        if pos == self.my_pos:
            self.hand[pos] = remove_cards(self.hand[pos], action)
        self.acting_pos = _NEXT_POS[pos]

    def is_my_turn(self):
        return self.acting_pos == self.my_pos

    def print_status(self):
        print()
        counts = self._num_cards_left()
        print('── 当前牌局状态 ──────────────────────────')
        for p in _POSITIONS:
            marker = ' ◀ 你' if p == self.my_pos else ''
            print(f'  {_POS_NAME[p]}({counts[p]}张){marker}')
        print(f'  炸弹数: {self.bomb_num}')
        last = self._get_last_move()
        print(f'  上一出牌: {cards2str(last) if last else "（无）"}')
        print(f'  你的手牌: {cards2str(self.hand[self.my_pos])}')
        print('──────────────────────────────────────────')


# ------------------------------------------------------------------ #
def load_model(position, path):
    from douzero.evaluation.deep_agent import _load_model
    return _load_model(position, path)


def input_cards(prompt):
    while True:
        try:
            raw = input(prompt).strip()
            return parse_cards(raw)
        except ValueError as e:
            print(f'  输入有误: {e}，请重新输入')
        except EOFError:
            sys.exit(0)


def ask_position():
    pos_map = {'1':'landlord','2':'landlord_down','3':'landlord_across','4':'landlord_up'}
    while True:
        print('你的位置：1=地主  2=下家  3=对面  4=上家')
        raw = input('> ').strip()
        if raw in pos_map:
            return pos_map[raw]
        print('  请输入 1/2/3/4')


def main():
    parser = argparse.ArgumentParser(description='DouZero 实战辅助（方案一：手动输入）')
    parser.add_argument('--model', type=str, default='',
                        help='模型 checkpoint 路径（可选，留空则仅列出合法动作）')
    args = parser.parse_args()

    print('=== DouZero 实战辅助 ===')
    print('牌面输入规则: 3 4 5 6 7 8 9 10 J Q K A 2 X(小王) D(大王)，空格分隔')
    print('过牌输入: 直接回车 或 输入 pass\n')

    my_pos = ask_position()
    my_hand = input_cards('输入你的手牌（空格分隔）: ')

    if my_pos == 'landlord':
        bottom = input_cards('输入底牌（8张）: ')
    else:
        bottom = input_cards('输入底牌（8张，已亮出）: ')

    model = None
    if args.model:
        print(f'加载模型: {args.model}')
        model = load_model(my_pos, args.model)
        print('模型加载成功')

    game = GameAssistant(my_pos, my_hand, bottom, model=model)

    print(f'\n游戏开始！你是【{_POS_NAME[my_pos]}】，先手为地主。\n')

    while True:
        pos = game.acting_pos

        if game.is_my_turn():
            game.print_status()
            best, ranked = game._ai_recommend()

            print(f'\n── 你的回合 ──')
            if model:
                print(f'  AI 推荐: 【{cards2str(best)}】')
                print('  候选动作（前5）:')
                for i, (score, m) in enumerate(ranked[:5]):
                    tag = ' ← 推荐' if i == 0 else ''
                    print(f'    {i+1}. {cards2str(m):<25} 评分: {score:.4f}{tag}')
            else:
                print('  合法动作:')
                for i, m in enumerate(ranked[:10]):
                    print(f'    {i+1}. {cards2str(m[1])}')

            action = input_cards('实际出牌（直接回车=PASS）: ')
            # Validate
            legal = game._get_legal_actions()
            if action not in legal:
                print('  ⚠ 该出牌不合法，请重新输入')
                continue
            game.apply_action(pos, action)
            if not action:
                print('  → 你选择 PASS')
            else:
                print(f'  → 你出牌: {cards2str(action)}')

            if len(game.hand[my_pos]) == 0:
                print('\n🎉 你出完了所有牌！游戏结束。')
                break

        else:
            counts = game._num_cards_left()
            name   = _POS_NAME[pos]
            print(f'\n── {name} 的回合（剩余{counts[pos]}张）──')
            last = game._get_last_move()
            if last:
                print(f'  需要压过: {cards2str(last)}')

            action = input_cards(f'输入 {name} 出的牌（直接回车=PASS）: ')
            game.apply_action(pos, action)

            remain = counts[pos] - len(action)
            if not action:
                print(f'  → {name} PASS（剩余{counts[pos]}张）')
            else:
                print(f'  → {name} 出牌: {cards2str(action)}（剩余{remain}张）')

            if remain == 0:
                winner = '地主' if pos == 'landlord' else '农民'
                print(f'\n游戏结束！{name} 出完手牌，{winner}获胜。')
                break


if __name__ == '__main__':
    main()
