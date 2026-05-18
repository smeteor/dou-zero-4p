"""Filter moves that can beat the rival's move (same type, higher rank, or higher bomb tier)."""
import collections
from douzero.env.utils import BOMB_TYPE_ORDER, TYPE_5_KING_BOMB


def _common_handle(moves, rival_move):
    return [m for m in moves if m[0] > rival_move[0]]


def filter_type_1_single(moves, rival_move):
    return _common_handle(moves, rival_move)


def filter_type_2_pair(moves, rival_move):
    return _common_handle(moves, rival_move)


def filter_type_3_triple(moves, rival_move):
    return _common_handle(moves, rival_move)


def filter_type_7_3_2(moves, rival_move):
    rival_rank = sorted(rival_move)[2]   # middle element = triple rank
    return [m for m in moves if sorted(m)[2] > rival_rank]


def filter_type_8_serial_single(moves, rival_move):
    return _common_handle(moves, rival_move)


def filter_type_9_serial_pair(moves, rival_move):
    return _common_handle(moves, rival_move)


def filter_type_10_serial_triple(moves, rival_move):
    return _common_handle(moves, rival_move)


def filter_type_12_serial_3_2(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = max(k for k, v in rival.items() if v == 3)
    result = []
    for m in moves:
        my = collections.Counter(m)
        my_rank = max(k for k, v in my.items() if v == 3)
        if my_rank > rival_rank:
            result.append(m)
    return result


def filter_bombs(my_bomb_moves, rival_move):
    """Return all bombs in my_bomb_moves that beat rival_move (also a bomb).
    Handles the full 6-tier hierarchy: 枪 < 炮 < 火箭 < 导弹 < 天炸 < 天尊.
    """
    from douzero.env.move_detector import get_move_type, get_bomb_tier
    rival_info = get_move_type(rival_move)
    rival_tier = get_bomb_tier(rival_info['type'])
    rival_rank = rival_info.get('rank', 0)

    result = []
    for m in my_bomb_moves:
        my_info = get_move_type(m)
        my_tier = get_bomb_tier(my_info['type'])
        if my_info['type'] == TYPE_5_KING_BOMB:
            result.append(m)
        elif my_tier > rival_tier:
            result.append(m)
        elif my_tier == rival_tier and my_info.get('rank', 0) > rival_rank:
            result.append(m)
    return result
