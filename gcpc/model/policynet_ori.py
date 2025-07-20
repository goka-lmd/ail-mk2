import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from rlf.policies.base_net_policy import BaseNetPolicy
from rlf.policies.base_policy import create_simple_action_data
import rlf.rl.utils as rutils
from gcpc.model.trajnet import TrajNet

from typing import Tuple, Union
from torch import device
Device = Union[device, str, int, None]

class SlotBasicPolicy(BaseNetPolicy):
    def __init__(self, is_stoch=False, fuse_states=[], use_goal=False, get_base_net_fn=None):
        super().__init__(use_goal, fuse_states, get_base_net_fn)
        self.state_norm_fn = lambda x: x
        self.action_denorm_fn = lambda x: x
        self.is_stoch = is_stoch

    def set_state_norm_fn(self, state_norm_fn):
        self.state_norm_fn = state_norm_fn

    def set_action_denorm_fn(self, action_denorm_fn):
        self.action_denorm_fn = action_denorm_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        mae_encoder = TrajNet.load_from_checkpoint(args.mae_ckpt_path, weights_only=False)
        self.slotmae = mae_encoder.model
        self.embed_dim = args.hidden_dim
        self.obs_dim = self.slotmae.obs_dim
        self.action_dim = self.slotmae.action_dim
        self.n_slots = self.slotmae.n_slots
        self.ctx_size = args.ctx_size

        for param in self.slotmae.parameters():
            param.requires_grad = False
        self.slotmae.eval()

        self.obs_embed = nn.Linear(self.obs_dim, self.embed_dim)
        self.bn_embed = nn.Linear(self.n_slots * self.slotmae.embed_dim, self.embed_dim)
        self.policy = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(args.pdrop),
            nn.Linear(self.embed_dim, self.action_dim)
        )

        if not rutils.is_discrete(self.action_space) and self.is_stoch:
            self.std_layer = nn.Linear(self.embed_dim, self.action_dim)


    def forward(self, observations, padding_mask, rnn_hxs, mask):
        batch_size = observations.shape[0]
        with torch.no_grad():
            latent_future, _ = self.slotmae.encode(observations, padding_mask)

        o_keep_list = []
        for i in range(batch_size):
            valid_o = observations[i][padding_mask[i] == 0]
            o_keep_list.append(valid_o)
        
        o_keep = pad_sequence(o_keep_list, batch_first=True)

        obs = F.relu(self.obs_embed(o_keep[:, -1:]))
        bottleneck = F.relu(self.bn_embed(latent_future.view(batch_size, 1, -1)))
        obs_bn = torch.cat([obs, bottleneck], dim=-1)
        pred = self.policy(obs_bn)

        if not rutils.is_discrete(self.action_space) and self.is_stoch:
            std = self.std_layer(obs_bn)
            dist = torch.distributions.Normal(pred, std)
            pred = dist.rsample()
        
        return pred, None, None

    def ar_mask(self, batch_size: int, length: int, keep_len: float, device: Device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def get_action(self, state, add_state, rnn_hxs, mask, step_info, past_obs):
        if past_obs.numel() == 0:
            state = state.unsqueeze(1)
        else:
            state = torch.cat([past_obs, state.unsqueeze(1)], dim=1)

        batch_size, length, _ = state.shape

        if length > self.ctx_size:
            state = state[:, -self.ctx_size:, :]
            length = self.ctx_size

        obs_mask = self.ar_mask(batch_size, length, length, state.device)

        with torch.no_grad():
            latent_future, _ = self.slotmae.encode(state, obs_mask)

        o_keep = state[obs_mask == 0].view(batch_size, -1, self.obs_dim)

        obs = F.relu(self.obs_embed(o_keep[:, -1:]))
        bottleneck = F.relu(self.bn_embed(latent_future.view(batch_size, 1, -1)))
        obs_bn = torch.cat([obs, bottleneck], dim=-1)

        ret_action = self.policy(obs_bn)

        if step_info.is_eval or not self.is_stoch:
            ret_action = rutils.get_ac_compact(self.action_space, ret_action)
        else:
            if rutils.is_discrete(self.action_space):
                dist = torch.distributions.Categorical(ret_action.softmax(dim=-1))
            else:
                std = self.std(obs_bn)
                dist = torch.distributions.Normal(ret_action, std)
            ret_action = dist.sample()

        return create_simple_action_data(ret_action, rnn_hxs)



    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--mae-ckpt-path", type=str, required=True)
        parser.add_argument("--pdrop", type=float, default=0)
        parser.add_argument("--ctx-size", type=int, default=4)