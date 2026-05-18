"""Neural network models for 4-player double-deck DouDizhu.

Observation dimensions:
  Landlord  x_no_action = 753  (6×108 + 3×25 + 30)
  Farmer    x_no_action = 869  (7×108 + 33 + 2×25 + 30)
  All       x_batch     = x_no_action + 108 (action)
  z shape   = (5, 432)  — 5 rounds × 4 players × 108-dim card vector
"""

import numpy as np
import torch
from torch import nn

_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']


class LandlordLstmModel(nn.Module):
    """Value network for the landlord position."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(432, 128, batch_first=True)
        # x_no_action(753) + action(108) = 861; + lstm_hidden(128) = 989
        self.dense1 = nn.Linear(861 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(x.shape[0], (1,))[0]
        else:
            action = torch.argmax(x, dim=0)[0]
        return dict(action=action)


class FarmerLstmModel(nn.Module):
    """Value network shared by all 3 farmer positions."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(432, 128, batch_first=True)
        # x_no_action(869) + action(108) = 977; + lstm_hidden(128) = 1105
        self.dense1 = nn.Linear(977 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(x.shape[0], (1,))[0]
        else:
            action = torch.argmax(x, dim=0)[0]
        return dict(action=action)


# Model class for each position (used for evaluation loading)
model_dict = {
    'landlord':        LandlordLstmModel,
    'landlord_down':   FarmerLstmModel,
    'landlord_across': FarmerLstmModel,
    'landlord_up':     FarmerLstmModel,
}


class Model:
    """Wrapper for the 4 position models."""

    def __init__(self, device=0):
        self.models = {}
        dev = 'cpu' if device == 'cpu' else f'cuda:{device}'
        self.models['landlord']        = LandlordLstmModel().to(torch.device(dev))
        self.models['landlord_down']   = FarmerLstmModel().to(torch.device(dev))
        self.models['landlord_across'] = FarmerLstmModel().to(torch.device(dev))
        self.models['landlord_up']     = FarmerLstmModel().to(torch.device(dev))

    def forward(self, position, z, x, training=False, flags=None):
        return self.models[position].forward(z, x, training, flags)

    def share_memory(self):
        for m in self.models.values():
            m.share_memory()

    def eval(self):
        for m in self.models.values():
            m.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
