import os
import typing
import logging
import traceback
import numpy as np
import time

import torch
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']

# x_no_action dims: landlord=753, farmers=869
_X_DIM = {'landlord': 753, 'landlord_down': 869,
           'landlord_across': 869, 'landlord_up': 869}


def create_env(flags):
    return Env(flags.objective)


def get_batch(free_queue, full_queue, buffers, flags, lock):
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def create_optimizers(flags, learner_model):
    optimizers = {}
    for position in _POSITIONS:
        optimizers[position] = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
    return optimizers


def create_buffers(flags, device_iterator):
    T = flags.unroll_length
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in _POSITIONS:
            x_dim = _X_DIM[position]
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 108), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 432), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    dev_str = 'cpu' if device == 'cpu' else f'cuda:{device}'
                    _buffer = torch.empty(**specs[key]).to(torch.device(dev_str)).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf             = {p: [] for p in _POSITIONS}
        episode_return_buf   = {p: [] for p in _POSITIONS}
        target_buf           = {p: [] for p in _POSITIONS}
        obs_x_no_action_buf  = {p: [] for p in _POSITIONS}
        obs_action_buf       = {p: [] for p in _POSITIONS}
        obs_z_buf            = {p: [] for p in _POSITIONS}
        size                 = {p: 0  for p in _POSITIONS}

        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output['done']:
                    for p in _POSITIONS:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False] * (diff - 1))
                            done_buf[p].append(True)
                            ep_ret = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0] * (diff - 1))
                            episode_return_buf[p].append(ep_ret)
                            target_buf[p].extend([ep_ret] * diff)
                    break

            for p in _POSITIONS:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...]             = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...]   = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...]           = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...]  = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...]       = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...]            = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p]            = done_buf[p][T:]
                    episode_return_buf[p]  = episode_return_buf[p][T:]
                    target_buf[p]          = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p]      = obs_action_buf[p][T:]
                    obs_z_buf[p]           = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        raise e


def _cards2tensor(list_cards):
    return torch.from_numpy(_cards2array(list_cards))
