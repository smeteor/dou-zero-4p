import itertools

# global parameters
MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

# action types
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4           # 枪: 4 same
TYPE_5_KING_BOMB = 5      # 天尊: all 4 jokers [20, 20, 30, 30]
TYPE_6_3_1 = 6            # NOT USED in 4-player (三带一单 invalid)
TYPE_7_3_2 = 7            # 三带一对
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11   # NOT USED in 4-player (飞机带单 invalid)
TYPE_12_SERIAL_3_2 = 12   # 飞机带连对 (consecutive pairs required)
TYPE_13_4_2 = 13          # NOT USED in 4-player
TYPE_14_4_22 = 14         # NOT USED in 4-player
TYPE_15_WRONG = 15
TYPE_16_PENTA_BOMB = 16   # 炮: 5 same
TYPE_17_HEXA_BOMB = 17    # 火箭/六喜: 6 same
TYPE_18_HEPTA_BOMB = 18   # 导弹/七巧: 7 same
TYPE_19_OCTA_BOMB = 19    # 天炸: 8 same

# Bomb types in ascending strength order (index = tier)
BOMB_TYPE_ORDER = [
    TYPE_4_BOMB,
    TYPE_16_PENTA_BOMB,
    TYPE_17_HEXA_BOMB,
    TYPE_18_HEPTA_BOMB,
    TYPE_19_OCTA_BOMB,
    TYPE_5_KING_BOMB,
]

# betting round action
PASS = 0
CALL = 1
RAISE = 2

# return all possible results of selecting num cards from cards list
def select(cards, num):
    return [list(i) for i in itertools.combinations(cards, num)]
