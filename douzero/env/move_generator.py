from douzero.env.utils import MIN_PAIRS, MIN_TRIPLES, select
import collections
import itertools

# Note: 顺子 (serial single) is NOT a valid move type in 4-player double-deck DouDiZhu.
# 2(17) and jokers (20, 30) cannot appear in any sequential combination.
# Sequence validity uses actual card values; A(14)-2(17) gap is 3 so they are
# naturally non-consecutive and will never appear together in generated moves.


class MovesGener(object):
    """Generate all valid move combinations from a hand (4-player double-deck rules)."""

    def __init__(self, cards_list):
        self.cards_list = cards_list
        self.cards_dict = collections.defaultdict(int)
        for c in cards_list:
            self.cards_dict[c] += 1

        self.single_card_moves = [];  self.gen_type_1_single()
        self.pair_moves = [];         self.gen_type_2_pair()
        self.triple_cards_moves = []; self.gen_type_3_triple()
        self.bomb_moves = [];         self.gen_type_4_bomb()
        self.penta_bomb_moves = [];   self.gen_type_penta_bomb()
        self.hexa_bomb_moves = [];    self.gen_type_hexa_bomb()
        self.hepta_bomb_moves = [];   self.gen_type_hepta_bomb()
        self.octa_bomb_moves = [];    self.gen_type_octa_bomb()
        self.king_bomb_moves = [];    self.gen_type_5_king_bomb()

    # ------------------------------------------------------------------
    # Serial move helper — uses actual card values (no seq-rank remapping).
    # Jokers (20, 30) and 2 (17) are excluded by callers before passing `cards`.
    # ------------------------------------------------------------------
    def _gen_serial_moves(self, cards, min_serial, repeat=1, repeat_num=0):
        if repeat_num < min_serial:
            repeat_num = 0

        unique_vals = sorted(set(cards))   # callers already filtered out 2/jokers
        seq_records = []
        start = i = 0
        longest = 1
        while i < len(unique_vals):
            if i + 1 < len(unique_vals) and unique_vals[i + 1] - unique_vals[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        moves = []
        for seq_start, seq_len in seq_records:
            if seq_len < min_serial:
                continue
            run = unique_vals[seq_start: seq_start + seq_len]
            if repeat_num == 0:
                steps = min_serial
                while steps <= seq_len:
                    idx = 0
                    while steps + idx <= seq_len:
                        moves.append(sorted(run[idx: idx + steps] * repeat))
                        idx += 1
                    steps += 1
            else:
                if seq_len < repeat_num:
                    continue
                idx = 0
                while idx + repeat_num <= seq_len:
                    moves.append(sorted(run[idx: idx + repeat_num] * repeat))
                    idx += 1
        return moves

    # ------------------------------------------------------------------
    # Basic types
    # ------------------------------------------------------------------
    def gen_type_1_single(self):
        self.single_card_moves = [[c] for c in set(self.cards_list)]
        return self.single_card_moves

    def gen_type_2_pair(self):
        self.pair_moves = [[k, k] for k, v in self.cards_dict.items() if v >= 2]
        return self.pair_moves

    def gen_type_3_triple(self):
        self.triple_cards_moves = [[k, k, k] for k, v in self.cards_dict.items() if v >= 3]
        return self.triple_cards_moves

    def gen_type_4_bomb(self):
        self.bomb_moves = [[k] * 4 for k, v in self.cards_dict.items() if v >= 4]
        return self.bomb_moves

    def gen_type_penta_bomb(self):
        self.penta_bomb_moves = [[k] * 5 for k, v in self.cards_dict.items() if v >= 5]
        return self.penta_bomb_moves

    def gen_type_hexa_bomb(self):
        self.hexa_bomb_moves = [[k] * 6 for k, v in self.cards_dict.items() if v >= 6]
        return self.hexa_bomb_moves

    def gen_type_hepta_bomb(self):
        self.hepta_bomb_moves = [[k] * 7 for k, v in self.cards_dict.items() if v >= 7]
        return self.hepta_bomb_moves

    def gen_type_octa_bomb(self):
        self.octa_bomb_moves = [[k] * 8 for k, v in self.cards_dict.items() if v >= 8]
        return self.octa_bomb_moves

    def gen_type_5_king_bomb(self):
        """天尊: requires 2 small jokers AND 2 large jokers."""
        self.king_bomb_moves = []
        if self.cards_dict.get(20, 0) >= 2 and self.cards_dict.get(30, 0) >= 2:
            self.king_bomb_moves.append([20, 20, 30, 30])
        return self.king_bomb_moves

    def gen_all_bombs(self):
        return (self.bomb_moves + self.penta_bomb_moves + self.hexa_bomb_moves +
                self.hepta_bomb_moves + self.octa_bomb_moves + self.king_bomb_moves)

    # ------------------------------------------------------------------
    # 三带一对 (triple + pair)
    # ------------------------------------------------------------------
    def gen_type_7_3_2(self):
        result = []
        for t in self.pair_moves:
            for tri in self.triple_cards_moves:
                if t[0] != tri[0]:
                    result.append(sorted(t + tri))
        return result

    # ------------------------------------------------------------------
    # 双顺 (consecutive pairs, 3+ pairs).
    # 2(17) is excluded; A(14)-2(17) gap = 3 so they won't join anyway.
    # ------------------------------------------------------------------
    def gen_type_9_serial_pair(self, repeat_num=0):
        # Only cards 3-A can form consecutive sequences (exclude 2 and jokers)
        pair_vals = [k for k, v in self.cards_dict.items()
                     if v >= 2 and k not in (17, 20, 30)]
        return self._gen_serial_moves(pair_vals, MIN_PAIRS, repeat=2, repeat_num=repeat_num)

    # ------------------------------------------------------------------
    # 三顺 / 飞机不带翅膀 (consecutive triples, 2+).
    # ------------------------------------------------------------------
    def gen_type_10_serial_triple(self, repeat_num=0):
        triple_vals = [k for k, v in self.cards_dict.items()
                       if v >= 3 and k not in (17, 20, 30)]
        return self._gen_serial_moves(triple_vals, MIN_TRIPLES, repeat=3, repeat_num=repeat_num)

    # ------------------------------------------------------------------
    # 飞机带翅膀: consecutive triples + same count of CONSECUTIVE pairs.
    # Wing pairs must also form a consecutive sequence and must not overlap
    # with the triple ranks.
    # ------------------------------------------------------------------
    def gen_type_12_serial_3_2(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        result = []

        # Pair candidates: exclude 2 and jokers (same sequential constraint)
        pair_pool = sorted(k for k, v in self.cards_dict.items()
                           if v >= 2 and k not in (17, 20, 30))

        for s3 in serial_3_moves:
            s3_ranks = set(s3)          # unique triple ranks
            s3_len   = len(s3_ranks)    # number of triple groups

            # Wing pairs must not overlap with triple ranks
            candidates = [k for k in pair_pool if k not in s3_ranks]

            # Find all consecutive sub-sequences of exactly s3_len from candidates
            for i in range(len(candidates) - s3_len + 1):
                subseq = candidates[i: i + s3_len]
                if subseq[-1] - subseq[0] == s3_len - 1:   # consecutive
                    result.append(sorted(s3 + [c for c in subseq for _ in range(2)]))

        return result

    # ------------------------------------------------------------------
    # All moves (no 顺子 / serial single in 4-player rules)
    # ------------------------------------------------------------------
    def gen_moves(self):
        moves = []
        moves.extend(self.gen_type_1_single())
        moves.extend(self.gen_type_2_pair())
        moves.extend(self.gen_type_3_triple())
        moves.extend(self.gen_type_4_bomb())
        moves.extend(self.gen_type_penta_bomb())
        moves.extend(self.gen_type_hexa_bomb())
        moves.extend(self.gen_type_hepta_bomb())
        moves.extend(self.gen_type_octa_bomb())
        moves.extend(self.gen_type_5_king_bomb())
        moves.extend(self.gen_type_7_3_2())
        moves.extend(self.gen_type_9_serial_pair())
        moves.extend(self.gen_type_10_serial_triple())
        moves.extend(self.gen_type_12_serial_3_2())
        return moves
