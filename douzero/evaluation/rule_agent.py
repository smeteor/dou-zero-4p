"""
Rule-based agent for 4-player double-deck DouDizhu.

Strategy summary:
  Leading  — decompose hand by move categories; play the combo that
             contains the smallest card (no card left behind).
  Following — match the rival's type with the smallest beating move;
             pass if the rival is a teammate (farmer cooperation);
             use a bomb when out of same-type moves and rival is enemy.

Adapted from the 3-player RLCard agent (rlcard_agent.py).  Main differences
are documented in the module docstring at the bottom of this file.
"""

import collections
import random

from douzero.env.move_detector import get_move_type
from douzero.env.utils import (
    TYPE_0_PASS,
    TYPE_1_SINGLE,
    TYPE_2_PAIR,
    TYPE_3_TRIPLE,
    TYPE_4_BOMB,
    TYPE_5_KING_BOMB,
    TYPE_7_3_2,
    TYPE_9_SERIAL_PAIR,
    TYPE_10_SERIAL_TRIPLE,
    TYPE_12_SERIAL_3_2,
    TYPE_16_PENTA_BOMB,
    TYPE_17_HEXA_BOMB,
    TYPE_18_HEPTA_BOMB,
    TYPE_19_OCTA_BOMB,
    BOMB_TYPE_ORDER,
)

# All bomb types (excluding TYPE_5 which is checked separately for overpower)
_BOMB_TYPES = {
    TYPE_4_BOMB, TYPE_16_PENTA_BOMB, TYPE_17_HEXA_BOMB,
    TYPE_18_HEPTA_BOMB, TYPE_19_OCTA_BOMB, TYPE_5_KING_BOMB,
}

# "Standard" non-bomb types that can be matched by same type
_STANDARD_TYPES = {
    TYPE_1_SINGLE, TYPE_2_PAIR, TYPE_3_TRIPLE,
    TYPE_7_3_2, TYPE_9_SERIAL_PAIR, TYPE_10_SERIAL_TRIPLE, TYPE_12_SERIAL_3_2,
}

# Teammate relations: farmers are all on the same team
_FARMERS = {'landlord_down', 'landlord_across', 'landlord_up'}


def _is_teammate(pos_a, pos_b):
    """Two positions are teammates if both are farmers."""
    return pos_a in _FARMERS and pos_b in _FARMERS


def _move_rank(move):
    """Return a comparable rank value for a move (used for smallest-beating)."""
    info = get_move_type(move)
    t = info['type']
    if t == TYPE_0_PASS:
        return 0
    if t in (TYPE_1_SINGLE, TYPE_2_PAIR, TYPE_3_TRIPLE):
        return info['rank']
    if t == TYPE_7_3_2:
        return info['rank']
    if t == TYPE_9_SERIAL_PAIR:
        return info['rank']
    if t == TYPE_10_SERIAL_TRIPLE:
        return info['rank']
    if t == TYPE_12_SERIAL_3_2:
        return info['rank']
    if t in _BOMB_TYPES:
        # Bomb tier × 100 + rank for ordering within same tier
        tier = BOMB_TYPE_ORDER.index(t) if t in BOMB_TYPE_ORDER else -1
        return tier * 100 + info.get('rank', 0)
    return 0


def _find_smallest_beating(legal_actions, rival_info):
    """Among legal_actions, return the weakest move that beats rival.
    Returns None if no matching-type move can beat.
    """
    rival_type = rival_info['type']
    rival_rank = rival_info.get('rank', 0)

    best = None
    best_rank = float('inf')

    for ac in legal_actions:
        if len(ac) == 0:
            continue
        my_info = get_move_type(ac)
        if my_info['type'] != rival_type:
            continue
        my_rank = my_info.get('rank', 0)

        if rival_type in (TYPE_1_SINGLE, TYPE_2_PAIR, TYPE_3_TRIPLE,
                          TYPE_7_3_2):
            if my_rank > rival_rank and my_rank < best_rank:
                best = ac
                best_rank = my_rank
        elif rival_type in (TYPE_9_SERIAL_PAIR, TYPE_10_SERIAL_TRIPLE,
                            TYPE_12_SERIAL_3_2):
            my_len = my_info.get('len', 0)
            rival_len = rival_info.get('len', 0)
            if my_len == rival_len and my_rank > rival_rank and my_rank < best_rank:
                best = ac
                best_rank = my_rank

    return best


class RuleAgent:
    """Rule-based agent for 4-player double-deck DouDizhu."""

    def __init__(self, position):
        self.position = position

    def act(self, infoset):
        legal_actions = infoset.legal_actions
        action = None

        try:
            last_move = infoset.last_move
            last_pid = infoset.last_pid

            # --- Leading: we played last and everyone else passed ---
            if last_pid == self.position:
                action = self._lead(infoset.player_hand_cards, legal_actions)
                if action is None:
                    # fall back: smallest single, or first legal action
                    singles = [a for a in legal_actions if len(a) == 1]
                    if singles:
                        action = min(singles, key=lambda a: a[0])
                    elif legal_actions:
                        action = legal_actions[0]

            # --- Following ---
            else:
                rival_info = get_move_type(last_move)

                # 1) Try to beat with the same type (smallest beating move)
                action = _find_smallest_beating(legal_actions, rival_info)

                # 2) Can't beat — if rival is teammate, pass
                if action is None and _is_teammate(self.position, last_pid):
                    if [] in legal_actions:
                        action = []

                # 3) Rival is enemy — try a bomb
                if action is None:
                    bomb_actions = [a for a in legal_actions if len(a) > 0
                                    and get_move_type(a)['type'] in _BOMB_TYPES]
                    if bomb_actions:
                        bomb_actions.sort(key=lambda a: _move_rank(a))
                        action = bomb_actions[0]

                # 4) Nothing to play — pass
                if action is None and [] in legal_actions:
                    action = []

        except Exception:
            import sys
            import traceback
            print(f'[RuleAgent {self.position}] exception in act():', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        # --- Validate and fallback ---
        if action is not None and action in legal_actions:
            return action

        # Debug: action validation failed
        import sys
        print(f'[RuleAgent {self.position}] validation failed:', file=sys.stderr)
        print(f'  action={action}', file=sys.stderr)
        print(f'  len(legal_actions)={len(legal_actions)}', file=sys.stderr)
        print(f'  last_pid={infoset.last_pid}', file=sys.stderr)
        print(f'  last_move={infoset.last_move}', file=sys.stderr)
        print(f'  hand_size={len(infoset.player_hand_cards)}', file=sys.stderr)
        if legal_actions:
            print(f'  legal_actions[0:3]={legal_actions[:3]}', file=sys.stderr)

        if legal_actions:
            return random.choice(legal_actions)
        return []

    # ------------------------------------------------------------------
    #  Leading: decompose hand and play the combo with the smallest card
    # ------------------------------------------------------------------
    def _lead(self, hand_cards, legal_actions):
        """Pick a leading move.  Decompose hand into categories, then
        return the combo that contains the smallest remaining card."""
        try:
            comb = _decompose_hand(hand_cards)
        except Exception:
            import sys
            import traceback
            print(f'[RuleAgent {self.position}] _decompose_hand failed:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return None

        # Walk through categories in order and pick the one containing
        # the minimum card value.
        categories = [
            'king_bomb',
            'bomb',
            'serial_3_2',
            'serial_triple',
            'serial_pair',
            'trio_2',
            'triple',
            'pair',
            'single',
        ]

        min_card = min(hand_cards)
        for cat_name in categories:
            moves = comb[cat_name]
            if not moves:
                continue
            for m in moves:
                if min_card in m:
                    # Verify this move is legal
                    if m in legal_actions:
                        return m
                    # If not directly legal, try a sorted copy
                    m_sorted = sorted(m)
                    if m_sorted in legal_actions:
                        return m_sorted

        return None


# ======================================================================
#  Hand decomposition for 4-player double-deck DouDizhu
# ======================================================================

def _decompose_hand(cards):
    """Decompose a hand (list of ints) into move categories.

    Extracts moves greedily in priority order: king bomb → bombs →
    serial_3_2 → serial_triple → serial_pair → trio_2 → triple → pair → single.

    Returns a dict with keys matching the category names above.
    """
    remaining = list(cards)
    counts = collections.Counter(remaining)

    comb = {
        'king_bomb':      [],
        'bomb':           [],
        'serial_3_2':     [],
        'serial_triple':  [],
        'serial_pair':    [],
        'trio_2':         [],
        'triple':         [],
        'pair':           [],
        'single':         [],
    }

    # --- 1. King bomb (天尊: 2 small jokers + 2 big jokers) ---
    if counts.get(20, 0) >= 2 and counts.get(30, 0) >= 2:
        comb['king_bomb'].append([20, 20, 30, 30])
        for c in [20, 20, 30, 30]:
            remaining.remove(c)
        counts = collections.Counter(remaining)

    # --- 2. Bombs (4-8 same rank, sorted by rank ascending so smaller
    #        bombs are consumed first, keeping larger ones for beating) ---
    for n in (4, 5, 6, 7, 8):
        for rank in sorted(c for c, cnt in counts.items() if cnt >= n):
            bomb = [rank] * n
            comb['bomb'].append(bomb)
            for _ in range(n):
                remaining.remove(rank)
        counts = collections.Counter(remaining)

    # --- 3. 飞机带连对 (serial triple + consecutive pairs) ---
    serial_3_2_moves = _extract_serial_3_2(remaining)
    for m in serial_3_2_moves:
        comb['serial_3_2'].append(m)
        for c in m:
            remaining.remove(c)
    counts = collections.Counter(remaining)

    # --- 4. 飞机 / serial triples (consecutive triples, 2+) ---
    serial_triple_moves = _extract_serial_triples(remaining)
    for m in serial_triple_moves:
        comb['serial_triple'].append(m)
        for c in m:
            remaining.remove(c)
    counts = collections.Counter(remaining)

    # --- 5. 连对 / serial pairs (consecutive pairs, 3+) ---
    serial_pair_moves = _extract_serial_pairs(remaining)
    for m in serial_pair_moves:
        comb['serial_pair'].append(m)
        for c in m:
            remaining.remove(c)
    counts = collections.Counter(remaining)

    # --- 6. 三带一对 ---
    triples = sorted(k for k, v in counts.items() if v >= 3)
    pairs_candidate = sorted(k for k, v in counts.items() if v >= 2)
    for t_rank in triples:
        for p_rank in pairs_candidate:
            if p_rank == t_rank:
                continue
            need = [t_rank] * 3 + [p_rank] * 2
            if all(remaining.count(c) >= need.count(c) for c in set(need)):
                comb['trio_2'].append(sorted(need))
                for c in need:
                    remaining.remove(c)
                break
        counts = collections.Counter(remaining)

    # --- 7. Remaining triples ---
    for rank in sorted(k for k, v in counts.items() if v >= 3):
        comb['triple'].append([rank] * 3)
        for _ in range(3):
            remaining.remove(rank)
    counts = collections.Counter(remaining)

    # --- 8. Remaining pairs ---
    for rank in sorted(k for k, v in counts.items() if v >= 2):
        need = (counts[rank] // 2) * 2
        for _ in range(need // 2):
            comb['pair'].append([rank, rank])
        for _ in range(need):
            remaining.remove(rank)
    counts = collections.Counter(remaining)

    # --- 9. Remaining singles ---
    for c in sorted(remaining):
        comb['single'].append([c])

    return comb


# ------------------------------------------------------------------
#  Serial extraction helpers (2-deck aware)
# ------------------------------------------------------------------

_SEQUENCE_RANKS = list(range(3, 15))   # 3–A, excluding 2(17) and jokers(20,30)
_RANK_INDEX = {r: i for i, r in enumerate(_SEQUENCE_RANKS)}  # 3→0 … A→11


def _get_seq_ranks(card_list):
    """Return sorted list of ranks in `card_list` that are sequence-eligible."""
    return sorted(set(c for c in card_list if c in _RANK_INDEX))


def _extract_serial_triples(cards):
    """Extract consecutive-triple moves (飞机, 2+ groups) from `cards`."""
    counts = collections.Counter(cards)
    seq_ranks = _get_seq_ranks(cards)

    result = []
    i = 0
    while i < len(seq_ranks):
        # Find longest consecutive run where each rank has ≥3 copies
        j = i
        while (j < len(seq_ranks) and
               (j == i or seq_ranks[j] - seq_ranks[j - 1] == 1) and
               counts[seq_ranks[j]] >= 3):
            j += 1
        run = seq_ranks[i:j]
        if len(run) >= 2:
            # Greedily take the whole run
            move = []
            for r in run:
                move.extend([r, r, r])
            result.append(sorted(move))
        i = j + 1 if j == i else j
    return result


def _extract_serial_pairs(cards):
    """Extract consecutive-pair moves (连对, 3+ pairs) from `cards`."""
    counts = collections.Counter(cards)
    seq_ranks = _get_seq_ranks(cards)

    result = []
    i = 0
    while i < len(seq_ranks):
        j = i
        while (j < len(seq_ranks) and
               (j == i or seq_ranks[j] - seq_ranks[j - 1] == 1) and
               counts[seq_ranks[j]] >= 2):
            j += 1
        run = seq_ranks[i:j]
        if len(run) >= 3:
            move = []
            for r in run:
                move.extend([r, r])
            result.append(sorted(move))
        i = j + 1 if j == i else j
    return result


def _extract_serial_3_2(cards):
    """Extract 飞机带连对 from `cards`.

    Finds consecutive triple runs, then for each run checks whether there
    exist enough consecutive-pair ranks (non-overlapping) to serve as wings.
    """
    counts = collections.Counter(cards)
    seq_ranks = _get_seq_ranks(cards)

    result = []

    # Find all consecutive triple runs
    triple_runs = []
    i = 0
    while i < len(seq_ranks):
        j = i
        while (j < len(seq_ranks) and
               (j == i or seq_ranks[j] - seq_ranks[j - 1] == 1) and
               counts[seq_ranks[j]] >= 3):
            j += 1
        run = seq_ranks[i:j]
        if len(run) >= 2:
            triple_runs.append(run)
        i = j + 1 if j == i else j

    for run in triple_runs:
        k = len(run)
        run_set = set(run)

        pair_candidates = [r for r in seq_ranks if r not in run_set and counts[r] >= 2]

        for start in range(len(pair_candidates) - k + 1):
            sub = pair_candidates[start:start + k]
            if all(sub[m + 1] - sub[m] == 1 for m in range(k - 1)):
                move = []
                for r in run:
                    move.extend([r, r, r])
                for r in sub:
                    move.extend([r, r])
                result.append(sorted(move))
                break
        if result:
            break

    return result


# ======================================================================
#  Differences from the 3-player RLCard agent (rlcard_agent.py)
# ======================================================================
#
#  1. Deck: 3P uses 54 cards (1 deck), 4P uses 108 cards (2 decks).
#     Each rank has up to 8 copies instead of 4.
#
#  2. Number of players: 3P = 1 landlord + 2 farmers; 4P = 1 landlord + 3
#     farmers.  Teammate detection is adapted accordingly.
#
#  3. Move types removed in 4P:
#     - 顺子 / serial single (TYPE_8): completely gone.
#     - 三带一单 (TYPE_6), 四带二 (TYPE_13/14): invalid.
#     - 火箭 / rocket (big+joker pair): replaced by 天尊 (all 4 jokers).
#
#  4. Move types added in 4P:
#     - 炮(5) / 火箭(6) / 导弹(7) / 天炸(8): higher-order bombs.
#     - 天尊 (king bomb): 2×small joker + 2×big joker = unbeatable.
#
#  5. 飞机带翅膀: 3P accepts arbitrary single/pair wings; 4P requires
#     CONSECUTIVE pairs (TYPE_12 only).
#
#  6. Dependencies: the 3P agent shells out to rlcard's CARD_TYPE for move
#     identification and relies on rlcard's DouDizhu game definitions.
#     The 4P agent uses this project's own move_detector module and
#     BOMB_TYPE_ORDER, making it self-contained.
#
#  7. Card representation: 3P agent converts between env integers and
#     rlcard's character encoding (T/J/Q/K/A/B/R).  4P agent works
#     natively with the env integer representation throughout.
#
#  8. Bomb strength: 3P has 2 tiers (normal bomb + rocket).  4P has
#     6 tiers (枪 < 炮 < 火箭 < 导弹 < 天炸 < 天尊), and within each
#     tier higher rank wins.
# ======================================================================
