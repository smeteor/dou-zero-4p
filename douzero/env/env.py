from collections import Counter
import numpy as np

from douzero.env.game import GameEnv

# Column index for each card value in the 8x13 matrix
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

# Up to 8 copies per rank in a 2-deck game
NumOnes2Array = {}
for _n in range(9):
    arr = np.zeros(8, dtype=np.int8)
    arr[:_n] = 1
    NumOnes2Array[_n] = arr

# 2-deck, 108-card deck
deck = []
for i in range(3, 15):
    deck.extend([i] * 8)      # 8 of each rank 3-A
deck.extend([17] * 8)         # 8 twos
deck.extend([20, 20, 30, 30]) # 2 small jokers + 2 large jokers

# Farmer teammate lookup: for each farmer position, (teammate1, teammate2)
_FARMER_TEAMMATES = {
    'landlord_down':   ('landlord_across', 'landlord_up'),
    'landlord_across': ('landlord_up',     'landlord_down'),
    'landlord_up':     ('landlord_down',   'landlord_across'),
}


class Env:
    """4-player double-deck DouDizhu multi-agent wrapper."""

    def __init__(self, objective):
        self.objective = objective
        self.players = {}
        for position in ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']:
            self.players[position] = DummyAgent(position)
        self._env = GameEnv(self.players)
        self.infoset = None

    def reset(self):
        self._env.reset()
        _deck = deck.copy()
        np.random.shuffle(_deck)
        # Each farmer gets 25 cards; landlord gets 25 + 8 bottom = 33
        card_play_data = {
            'landlord':        _deck[:25] + _deck[100:108],  # 25 hand + 8 bottom
            'landlord_down':   _deck[25:50],
            'landlord_across': _deck[50:75],
            'landlord_up':     _deck[75:100],
            'eight_landlord_cards': _deck[100:108],
        }
        for key in card_play_data:
            card_play_data[key].sort()
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset
        return get_obs(self.infoset)

    def step(self, action):
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 1.0
            elif self.objective == 'logadp':
                return 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -1.0
            elif self.objective == 'logadp':
                return -1.0
            else:
                return -1.0

    @property
    def _game_infoset(self):
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        return self._env.acting_player_position

    @property
    def _game_over(self):
        return self._env.game_over


class DummyAgent(object):
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        self.action = action


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def get_obs(infoset):
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    else:
        return _get_obs_farmer(infoset)


def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[min(num_left_cards, max_num_cards) - 1] = 1
    return one_hot


def _cards2array(list_cards):
    """Encode a list of card values into a 108-dim vector.
    Layout: 8×13 column-major matrix (104 dims) + 4 joker bits.
    Joker bits: [X_copy1, X_copy2, D_copy1, D_copy2].
    """
    if len(list_cards) == 0:
        return np.zeros(108, dtype=np.int8)
    matrix = np.zeros([8, 13], dtype=np.int8)
    jokers = np.zeros(4, dtype=np.int8)
    counter = Counter(list_cards)
    for card, cnt in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[cnt]
        elif card == 20:
            jokers[0] = 1
            if cnt >= 2:
                jokers[1] = 1
        elif card == 30:
            jokers[2] = 1
            if cnt >= 2:
                jokers[3] = 1
    return np.concatenate((matrix.flatten('F'), jokers))


def _action_seq_list2array(action_seq_list):
    """Encode the last 20 actions (5 rounds × 4 players) into (5, 432)."""
    action_seq_array = np.zeros((len(action_seq_list), 108))
    for row, cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(cards)
    return action_seq_array.reshape(5, 432)


def _process_action_seq(sequence, length=20):
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        pad = [[] for _ in range(length - len(sequence))]
        pad.extend(sequence)
        sequence = pad
    return sequence


def _get_one_hot_bomb(bomb_num):
    one_hot = np.zeros(30)
    one_hot[min(bomb_num, 29)] = 1
    return one_hot


# ---------------------------------------------------------------------------
# Landlord observation
# x_no_action = 6×108 + 3×25 + 30 = 648 + 75 + 30 = 753
# x_batch      = 753 + 108 = 861
# ---------------------------------------------------------------------------

def _get_obs_landlord(infoset):
    num_actions = len(infoset.legal_actions)

    def _batch(arr):
        return np.repeat(arr[np.newaxis, :], num_actions, axis=0)

    my_hand        = _cards2array(infoset.player_hand_cards)
    other_hand     = _cards2array(infoset.other_hand_cards)
    last_action    = _cards2array(infoset.last_move)
    down_played    = _cards2array(infoset.played_cards['landlord_down'])
    across_played  = _cards2array(infoset.played_cards['landlord_across'])
    up_played      = _cards2array(infoset.played_cards['landlord_up'])
    down_left      = _get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 25)
    across_left    = _get_one_hot_array(infoset.num_cards_left_dict['landlord_across'], 25)
    up_left        = _get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 25)
    bomb_num       = _get_one_hot_bomb(infoset.bomb_num)

    x_no_action = np.hstack((my_hand, other_hand, last_action,
                              down_played, across_played, up_played,
                              down_left, across_left, up_left, bomb_num))

    action_arr = np.zeros((num_actions, 108), dtype=np.int8)
    for j, act in enumerate(infoset.legal_actions):
        action_arr[j] = _cards2array(act)

    x_batch = np.hstack((_batch(my_hand), _batch(other_hand), _batch(last_action),
                          _batch(down_played), _batch(across_played), _batch(up_played),
                          _batch(down_left), _batch(across_left), _batch(up_left),
                          _batch(bomb_num), action_arr))

    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], num_actions, axis=0)

    return {
        'position': 'landlord',
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }


# ---------------------------------------------------------------------------
# Farmer observation (shared architecture for all 3 farmer positions)
# x_no_action = 7×108 + 33 + 2×25 + 30 = 756 + 113 = 869
# x_batch      = 869 + 108 = 977
# ---------------------------------------------------------------------------

def _get_obs_farmer(infoset):
    pos = infoset.player_position
    t1, t2 = _FARMER_TEAMMATES[pos]
    num_actions = len(infoset.legal_actions)

    def _batch(arr):
        return np.repeat(arr[np.newaxis, :], num_actions, axis=0)

    my_hand           = _cards2array(infoset.player_hand_cards)
    other_hand        = _cards2array(infoset.other_hand_cards)
    landlord_played   = _cards2array(infoset.played_cards['landlord'])
    t1_played         = _cards2array(infoset.played_cards[t1])
    t2_played         = _cards2array(infoset.played_cards[t2])
    last_action       = _cards2array(infoset.last_move)
    last_landlord_act = _cards2array(infoset.last_move_dict['landlord'])
    landlord_left     = _get_one_hot_array(infoset.num_cards_left_dict['landlord'], 33)
    t1_left           = _get_one_hot_array(infoset.num_cards_left_dict[t1], 25)
    t2_left           = _get_one_hot_array(infoset.num_cards_left_dict[t2], 25)
    bomb_num          = _get_one_hot_bomb(infoset.bomb_num)

    x_no_action = np.hstack((my_hand, other_hand,
                              landlord_played, t1_played, t2_played,
                              last_action, last_landlord_act,
                              landlord_left, t1_left, t2_left, bomb_num))

    action_arr = np.zeros((num_actions, 108), dtype=np.int8)
    for j, act in enumerate(infoset.legal_actions):
        action_arr[j] = _cards2array(act)

    x_batch = np.hstack((_batch(my_hand), _batch(other_hand),
                          _batch(landlord_played), _batch(t1_played), _batch(t2_played),
                          _batch(last_action), _batch(last_landlord_act),
                          _batch(landlord_left), _batch(t1_left), _batch(t2_left),
                          _batch(bomb_num), action_arr))

    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], num_actions, axis=0)

    return {
        'position': pos,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
