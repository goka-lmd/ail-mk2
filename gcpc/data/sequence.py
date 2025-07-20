import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, default_convert, default_collate
from typing import Optional, TypeVar
from collections import namedtuple
import numpy as np
import random
import pytorch_lightning as pl
from typing import Dict
from omegaconf import DictConfig

T = TypeVar('T')
Batch = namedtuple('Batch', ['observations', 'valid_length', 'padding_mask'])


def build_buffer(flat_obs: torch.Tensor, done_flags: torch.Tensor, used_length):
    
    done_indices = torch.nonzero(done_flags).squeeze(-1).tolist()
    done_indices = [int(i) for i in done_indices]
    if used_length == 0:
        start_indices = [0] + [i + 1 for i in done_indices[:-1]]
    else:
        start_indices = [used_length + 1] + [i + 1 for i in done_indices[:-1]]
    episode_lengths = [end - start for start, end in zip(start_indices, done_indices)]

    max_len = max(episode_lengths)
    obs_dim = flat_obs.shape[1]
    num_episodes = len(episode_lengths)

    padded_obs = torch.zeros((num_episodes, max_len, obs_dim), dtype=flat_obs.dtype)
    for i, (start, end) in enumerate(zip(start_indices, done_indices)):
        episode = flat_obs[start:end]
        padded_obs[i, :len(episode)] = episode
        used_length = end

    buffer = {
        'obs': padded_obs,
        'episode_lengths': np.array(episode_lengths)
    }
    return buffer, used_length


class SequenceDataset(Dataset):
    def __init__(self, env_name: str, epi_buffers: Dict, ctx_size: int = 100):
        super().__init__()
        self.env_name = env_name
        self.ctx_size = ctx_size  # context window size
        self.epi_buffers = epi_buffers
        self.traj_indices = self.sample_trajs_from_episode()

    def sample_trajs_from_episode(self):
        '''
            makes indices for sampling from dataset;
            each index maps to a trajectory (start, end)
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                end = start + self.ctx_size
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.traj_indices)

    def __getitem__(self, idx: int):
        epi_idx, start, end = self.traj_indices[idx]

        epi_length = self.epi_buffers['episode_lengths'][epi_idx]
        observations = self.epi_buffers['obs'][epi_idx, start:end]

        valid_length = min(epi_length, end) - start

        padding_mask = np.zeros(shape=(self.ctx_size, ), dtype=np.bool8)  # only consider one modality length
        padding_mask[valid_length:] = True

        batch = Batch(observations, valid_length, padding_mask)

        return batch


class PretrainDataset(SequenceDataset):
    def sample_trajs_from_episode(self):
        '''
            makes start indices for sampling from dataset
        '''
        indices = []
        for i, epi_length in enumerate(self.epi_buffers['episode_lengths']):
            max_start = epi_length - self.ctx_size
            for start in range(max_start):
                indices.append((i, start))
        indices = np.array(indices)
        return indices

    def __getitem__(self, idx: int):
        epi_idx, start = self.traj_indices[idx]
        epi_length = self.epi_buffers['episode_lengths'][epi_idx]

        end = random.choice(range(start + self.ctx_size, epi_length))
        valid_length = end - start

        observations = self.epi_buffers['obs'][epi_idx, start:end]

        return observations, valid_length


def pt_collate_fn(batch):
    batch_size = len(batch)
    obss = default_convert([item[0] for item in batch])
    valid_lengths = [item[1] for item in batch]

    max_valid_length = max(valid_lengths)
    pad_observations = torch.zeros(batch_size, max_valid_length, obss[0].shape[-1])
    padding_mask = torch.zeros(batch_size, max_valid_length, dtype=torch.bool)
    for idx, item in enumerate(zip(obss, valid_lengths)):
        obs, valid_len = item
        pad_observations[idx, :valid_len] = obs
        padding_mask[idx, valid_len:] = True
    valid_lengths = torch.tensor(valid_lengths)
    batch = Batch(pad_observations, valid_lengths, padding_mask)
    return batch


class MAEDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.env_name = config.env_name
        self.batch_size = config.exp.batch_size
        self.ctx_size = config.exp.ctx_size
        self.num_workers = config.num_workers
        self.train_size = config.train_size
        self.dataset_path = config.traj_load_path

    def setup(self, stage: str):
        epi_buffer = torch.load(self.dataset_path)

        self._obs_dim = epi_buffer['obs'].shape[-1]
        self._action_dim = epi_buffer['actions'].shape[-1]
        done_indices = torch.nonzero(epi_buffer['done']).squeeze()
        num_epis = len(done_indices)
        
        if self.train_size <= 1:
            num_train = int(num_epis * self.train_size)
        else:
            num_train = self.train_size

        train_mask = torch.zeros_like(epi_buffer['done'], dtype=torch.bool)
        train_mask[done_indices[:num_train]] = True
        val_mask = torch.zeros_like(epi_buffer['done'], dtype=torch.bool)
        val_mask[done_indices[num_train:]] = True

        train_buffer, used_length = build_buffer(epi_buffer['obs'], train_mask, 0)
        val_buffer, used_length = build_buffer(epi_buffer['obs'], val_mask, used_length)

        if stage == 'trl':  # trajectory representation learning
            self.train = PretrainDataset(self.env_name, train_buffer, self.ctx_size)
            self.val = PretrainDataset(self.env_name, val_buffer, self.ctx_size)
        else:
            self.train = SequenceDataset(self.env_name, train_buffer, self.ctx_size)
            self.val = SequenceDataset(self.env_name, val_buffer, self.ctx_size)


    def train_dataloader(self):
        if isinstance(self.train, PretrainDataset):
            return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.val, PretrainDataset):
            return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, collate_fn=pt_collate_fn, pin_memory=True)
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def get_obs_dim(self):
        return self._obs_dim
    
    def get_action_dim(self):
        return self._action_dim
