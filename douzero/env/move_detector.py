from douzero.env.utils import *
import collections


def is_continuous_seq(ranks):
    """Check if a sorted list of card values forms a consecutive sequence.
    Jokers (20, 30) and 2 (17) are excluded — they cannot be in sequences.
    Returns False immediately if any such card is present.
    """
    if any(c in (17, 20, 30) for c in ranks):
        return False
    return all(ranks[i + 1] - ranks[i] == 1 for i in range(len(ranks) - 1))


def _is_ranks_continuous(ranks):
    """Check that a sorted list of integers is consecutive (diff=1 each step)."""
    return all(ranks[i + 1] - ranks[i] == 1 for i in range(len(ranks) - 1))


def get_move_type(move):
    move_size = len(move)
    move_dict = collections.Counter(move)

    if move_size == 0:
        return {'type': TYPE_0_PASS}

    if move_size == 1:
        return {'type': TYPE_1_SINGLE, 'rank': move[0]}

    if move_size == 2:
        if move[0] == move[1]:
            return {'type': TYPE_2_PAIR, 'rank': move[0]}
        return {'type': TYPE_15_WRONG}

    if move_size == 3:
        if len(move_dict) == 1:
            return {'type': TYPE_3_TRIPLE, 'rank': move[0]}
        return {'type': TYPE_15_WRONG}

    # 天尊: all 4 jokers [20, 20, 30, 30]
    if sorted(move) == [20, 20, 30, 30]:
        return {'type': TYPE_5_KING_BOMB}

    # Pure-same-rank bombs (枪4, 炮5, 火箭6, 导弹7, 天炸8)
    if len(move_dict) == 1:
        rank = list(move_dict.keys())[0]
        if move_size == 4:
            return {'type': TYPE_4_BOMB, 'rank': rank}
        if move_size == 5:
            return {'type': TYPE_16_PENTA_BOMB, 'rank': rank}
        if move_size == 6:
            return {'type': TYPE_17_HEXA_BOMB, 'rank': rank}
        if move_size == 7:
            return {'type': TYPE_18_HEPTA_BOMB, 'rank': rank}
        if move_size == 8:
            return {'type': TYPE_19_OCTA_BOMB, 'rank': rank}
        return {'type': TYPE_15_WRONG}

    if move_size == 5:
        # 三带一对 (3+pair): the only valid 5-card non-bomb in 4-player
        if len(move_dict) == 2:
            counts = sorted(move_dict.values())
            if counts == [2, 3]:
                rank = max(k for k, v in move_dict.items() if v == 3)
                return {'type': TYPE_7_3_2, 'rank': rank}
        return {'type': TYPE_15_WRONG}

    count_dict = collections.defaultdict(int)
    for c, n in move_dict.items():
        count_dict[n] += 1

    mdkeys = sorted(move_dict.keys())

    # Serial pair (双顺: 3+ consecutive pairs, no 2/jokers in sequence)
    # 2(17) cannot follow A(14) since 17-14=3 ≠ 1, so it is naturally excluded.
    if count_dict.get(2) == len(move_dict) and len(mdkeys) >= MIN_PAIRS:
        if is_continuous_seq(mdkeys):
            return {'type': TYPE_9_SERIAL_PAIR, 'rank': mdkeys[0], 'len': len(mdkeys)}

    # Serial triple / 飞机不带翅膀 (2+ consecutive triples, no 2/jokers)
    if count_dict.get(3) == len(move_dict) and len(mdkeys) >= MIN_TRIPLES:
        if is_continuous_seq(mdkeys):
            return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': mdkeys[0], 'len': len(mdkeys)}

    # 飞机带翅膀: consecutive triples + same count of consecutive pairs
    # Both the triple-ranks and the pair-ranks must form consecutive sequences.
    if count_dict.get(3, 0) >= MIN_TRIPLES and count_dict.get(2, 0) >= MIN_TRIPLES:
        if set(count_dict.keys()) == {2, 3}:
            serial_3_ranks = sorted(k for k, v in move_dict.items() if v == 3)
            pair_ranks     = sorted(k for k, v in move_dict.items() if v == 2)
            if (len(serial_3_ranks) == len(pair_ranks) and
                    is_continuous_seq(serial_3_ranks) and
                    is_continuous_seq(pair_ranks)):
                return {'type': TYPE_12_SERIAL_3_2,
                        'rank': serial_3_ranks[0],
                        'len': len(serial_3_ranks)}

    return {'type': TYPE_15_WRONG}


def get_bomb_tier(move_type_id):
    """Return bomb strength tier (higher = stronger). -1 if not a bomb."""
    try:
        return BOMB_TYPE_ORDER.index(move_type_id)
    except ValueError:
        return -1
