import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act

_POSITIONS = ['landlord', 'landlord_down', 'landlord_across', 'landlord_up']

mean_episode_return_buf = {p: deque(maxlen=100) for p in _POSITIONS}


def compute_loss(logits, targets):
    return ((logits.squeeze(-1) - targets) ** 2).mean()


def learn(position, actor_models, model, batch, optimizer, flags, lock):
    if flags.training_device != 'cpu':
        device = torch.device(f'cuda:{flags.training_device}')
    else:
        device = torch.device('cpu')

    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action      = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            f'mean_episode_return_{position}': torch.mean(
                torch.stack([r for r in mean_episode_return_buf[position]])).item(),
            f'loss_{position}': loss.item(),
        }
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats


def train(flags):
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. Train with CPU: "
                "python train.py --actor_device_cpu --training_device cpu")

    plogger = FileWriter(xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir)
    checkpointpath = os.path.expandvars(
        os.path.expanduser(f'{flags.savedir}/{flags.xpid}/model.tar'))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), \
            'num_actor_devices cannot exceed available GPU count'

    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    buffers = create_buffers(flags, device_iterator)

    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        free_queue[device] = {p: ctx.SimpleQueue() for p in _POSITIONS}
        full_queue[device] = {p: ctx.SimpleQueue() for p in _POSITIONS}

    learner_model = Model(device=flags.training_device)
    optimizers = create_optimizers(flags, learner_model)

    stat_keys = []
    for p in _POSITIONS:
        stat_keys += [f'mean_episode_return_{p}', f'loss_{p}']
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {p: 0 for p in _POSITIONS}

    if flags.load_model and os.path.exists(checkpointpath):
        dev_str = f'cuda:{flags.training_device}' if flags.training_device != 'cpu' else 'cpu'
        checkpoint_states = torch.load(checkpointpath, map_location=dev_str)
        for k in _POSITIONS:
            learner_model.get_model(k).load_state_dict(checkpoint_states['model_state_dict'][k])
            optimizers[k].load_state_dict(checkpoint_states['optimizer_state_dict'][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states['stats']
        frames = checkpoint_states['frames']
        position_frames = checkpoint_states['position_frames']
        log.info(f'Resuming preempted job, current stats:\n{stats}')

    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device],
                      models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock,
                        lock=threading.Lock()):
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position],
                              full_queue[device][position],
                              buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position),
                           batch, optimizers[position], flags, position_lock)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            for p in _POSITIONS:
                free_queue[device][p].put(m)

    threads = []
    locks = {device: {p: threading.Lock() for p in _POSITIONS} for device in device_iterator}
    position_locks = {p: threading.Lock() for p in _POSITIONS}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in _POSITIONS:
                thread = threading.Thread(
                    target=batch_and_learn,
                    name=f'batch-and-learn-{i}',
                    args=(i, device, position,
                          locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            'stats': stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames,
        }, checkpointpath)
        for position in _POSITIONS:
            weights_path = os.path.expandvars(os.path.expanduser(
                f'{flags.savedir}/{flags.xpid}/{position}_weights_{frames}.ckpt'))
            torch.save(learner_model.get_model(position).state_dict(), weights_path)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k: (position_frames[k] - position_start_frames[k]) /
                            (end_time - start_time) for k in position_frames}
            log.info(
                'After %i (L:%i D:%i A:%i U:%i) frames: @ %.1f fps (avg@ %.1f fps) Stats:\n%s',
                frames,
                position_frames['landlord'],
                position_frames['landlord_down'],
                position_frames['landlord_across'],
                position_frames['landlord_up'],
                fps, fps_avg,
                pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
