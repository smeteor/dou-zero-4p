from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

# Four-player, 2-deck: all bomb combinations that trigger bomb_num increment.
# Regular card values (3-A and 2)
_REGULAR = list(range(3, 15)) + [17]
bombs = []
for _n in range(4, 9):       # 枪(4), 炮(5), 火箭(6), 导弹(7), 天炸(8)
    for _v in _REGULAR:
        bombs.append([_v] * _n)
bombs.append([20, 20, 30, 30])  # 天尊

# Turn order: landlord → landlord_down → landlord_across → landlord_up → repeat
_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']
_NEXT_POS = {p: _POSITIONS[(_POSITIONS.index(p) + 1) % 4] for p in _POSITIONS}

# For each position: which positions are "farmers" (non-landlord)?
_FARMERS = [p for p in _POSITIONS if p != 'landlord']


class GameEnv(object):

    def __init__(self, players):
        self.card_play_action_seq = []
        self.eight_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_utility_dict = None

        self.players = players

        self.last_move_dict = {p: [] for p in _POSITIONS}
        self.played_cards = {p: [] for p in _POSITIONS}
        self.last_move = []
        self.last_two_moves = []

        self.num_wins = {'landlord': 0, 'farmer': 0}
        self.num_scores = {'landlord': 0, 'farmer': 0}

        self.info_sets = {p: InfoSet(p) for p in _POSITIONS}

        self.bomb_num = 0
        self.last_pid = 'landlord'

    def card_play_init(self, card_play_data):
        for p in _POSITIONS:
            self.info_sets[p].player_hand_cards = card_play_data[p]
        self.eight_landlord_cards = card_play_data['eight_landlord_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        for p in _POSITIONS:
            if len(self.info_sets[p].player_hand_cards) == 0:
                self.compute_player_utility(p)
                self.update_num_wins_scores()
                self.game_over = True
                return

    def compute_player_utility(self, winner_pos):
        if winner_pos == 'landlord':
            self.player_utility_dict = {'landlord': 3, 'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -3, 'farmer': 1}

    def update_num_wins_scores(self):
        for pos, utility in self.player_utility_dict.items():
            base_score = 3 if pos == 'landlord' else 1
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                self.num_scores[pos] += base_score
            else:
                self.num_scores[pos] -= base_score

    def get_winner(self):
        return self.winner

    def get_bomb_num(self):
        return self.bomb_num

    def step(self):
        action = self.players[self.acting_player_position].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if len(action) > 0:
            self.last_pid = self.acting_player_position

        if action in bombs:
            self.bomb_num += 1

        self.last_move_dict[self.acting_player_position] = action.copy()
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)
        self.played_cards[self.acting_player_position] += action

        # Track which bottom cards the landlord has played
        if self.acting_player_position == 'landlord' and len(action) > 0:
            for card in action:
                if card in self.eight_landlord_cards:
                    self.eight_landlord_cards.remove(card)

        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        """Return the most recent non-pass move."""
        for move in reversed(self.card_play_action_seq):
            if len(move) > 0:
                return move
        return []

    def get_last_two_moves(self):
        last_two = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two.insert(0, card)
            last_two = last_two[:2]
        return last_two

    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'
        else:
            self.acting_player_position = _NEXT_POS[self.acting_player_position]
        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action:
            for card in action:
                self.info_sets[self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        mg = MovesGener(
            self.info_sets[self.acting_player_position].player_hand_cards)

        # If the last non-pass move belongs to us, all others passed → play freely
        if self.last_pid == self.acting_player_position:
            rival_move = []
        else:
            rival_move = self.get_last_move()

        rival_info = md.get_move_type(rival_move)
        rival_move_type = rival_info['type']
        rival_move_len = rival_info.get('len', 1)
        moves = []

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            moves = ms.filter_type_1_single(mg.gen_type_1_single(), rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            moves = ms.filter_type_2_pair(mg.gen_type_2_pair(), rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            moves = ms.filter_type_3_triple(mg.gen_type_3_triple(), rival_move)

        elif rival_move_type == md.TYPE_7_3_2:
            moves = ms.filter_type_7_3_2(mg.gen_type_7_3_2(), rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type in (md.TYPE_4_BOMB, md.TYPE_16_PENTA_BOMB,
                                  md.TYPE_17_HEXA_BOMB, md.TYPE_18_HEPTA_BOMB,
                                  md.TYPE_19_OCTA_BOMB):
            # Can only beat with a stronger bomb
            moves = ms.filter_bombs(mg.gen_all_bombs(), rival_move)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []  # Nothing beats 天尊

        # Non-bomb rival: player may also play any bomb
        _bomb_types = {md.TYPE_4_BOMB, md.TYPE_16_PENTA_BOMB, md.TYPE_17_HEXA_BOMB,
                       md.TYPE_18_HEPTA_BOMB, md.TYPE_19_OCTA_BOMB, md.TYPE_5_KING_BOMB}
        if rival_move_type not in _bomb_types:
            moves = moves + mg.gen_all_bombs()

        # May always pass (unless it's our own turn to open)
        if len(rival_move) != 0:
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def reset(self):
        self.card_play_action_seq = []
        self.eight_landlord_cards = None
        self.game_over = False
        self.acting_player_position = None
        self.player_utility_dict = None

        self.last_move_dict = {p: [] for p in _POSITIONS}
        self.played_cards = {p: [] for p in _POSITIONS}
        self.last_move = []
        self.last_two_moves = []
        self.info_sets = {p: InfoSet(p) for p in _POSITIONS}
        self.bomb_num = 0
        self.last_pid = 'landlord'

    def get_infoset(self):
        pos = self.acting_player_position

        self.info_sets[pos].last_pid = self.last_pid
        self.info_sets[pos].legal_actions = self.get_legal_card_play_actions()
        self.info_sets[pos].bomb_num = self.bomb_num
        self.info_sets[pos].last_move = self.get_last_move()
        self.info_sets[pos].last_two_moves = self.get_last_two_moves()
        self.info_sets[pos].last_move_dict = self.last_move_dict
        self.info_sets[pos].num_cards_left_dict = {
            p: len(self.info_sets[p].player_hand_cards) for p in _POSITIONS}
        self.info_sets[pos].other_hand_cards = []
        for p in _POSITIONS:
            if p != pos:
                self.info_sets[pos].other_hand_cards += self.info_sets[p].player_hand_cards
        self.info_sets[pos].played_cards = self.played_cards
        self.info_sets[pos].eight_landlord_cards = self.eight_landlord_cards
        self.info_sets[pos].card_play_action_seq = self.card_play_action_seq
        self.info_sets[pos].all_handcards = {
            p: self.info_sets[p].player_hand_cards for p in _POSITIONS}

        return deepcopy(self.info_sets[pos])


class InfoSet(object):
    """Information set for one player in the 4-player game."""

    def __init__(self, player_position):
        self.player_position = player_position
        self.player_hand_cards = None
        self.num_cards_left_dict = None
        self.eight_landlord_cards = None
        self.card_play_action_seq = None
        self.other_hand_cards = None
        self.legal_actions = None
        self.last_move = None
        self.last_two_moves = None
        self.last_move_dict = None
        self.played_cards = None
        self.all_handcards = None
        self.last_pid = None
        self.bomb_num = None
