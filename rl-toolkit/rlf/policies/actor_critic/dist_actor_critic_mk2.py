import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from rlf.policies.base_policy import ActionData
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.actor_critic.base_actor_critic import ActorCritic
from ail_mk2.model.trajnet import TrajNet


class DistActorCritic_mk2(ActorCritic):
    """
    Defines an actor/critic where the actor outputs an action distribution
    """

    def __init__(self,
                get_actor_fn=None,
                get_dist_fn=None,
                get_critic_fn=None,
                get_critic_head_fn=None,
                fuse_states=[],
                use_goal=False,
                get_base_net_fn=None):
        super().__init__(get_critic_fn, get_critic_head_fn, use_goal,
                fuse_states, get_base_net_fn)
        """
        -   get_actor_fn: (obs_space : (int), input_shape : (int) ->
            rlf.rl.model.BaseNet)
        """

        if get_actor_fn is None:
            get_actor_fn = putils.get_def_actor
        self.get_actor_fn = get_actor_fn
        if get_dist_fn is None:
            get_dist_fn = putils.get_def_dist
        self.get_dist_fn = get_dist_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        mae_encoder = TrajNet.load_from_checkpoint(args.mae_ckpt_path, weights_only=False)
        self.slotmae = mae_encoder.model
        # self.embed_dim = args.hidden_dim
        self.obs_dim = self.slotmae.obs_dim
        self.action_dim = self.slotmae.action_dim
        self.n_slots = self.slotmae.n_slots
        self.ctx_size = args.ctx_size
        self.actor = self.get_actor_fn(
            rutils.get_obs_shape(obs_space, args.policy_ob_key)[0]*2,
            self._get_base_out_shape())
        self.dist = self.get_dist_fn(
            self.actor.output_shape, self.action_space)

        for param in self.slotmae.parameters():
            param.requires_grad = False

        self.slotmae.eval().to(args.device)

    def ar_mask(self, batch_size: int, length: int, keep_lens: torch.Tensor, device):
        keep_lens = keep_lens.to(device).long()
        idxs = torch.arange(length, device=device).unsqueeze(0)
        mask = (idxs >= keep_lens.unsqueeze(1)).float()
        return mask

    def get_action(self, state, add_state, hxs, masks, step_info, past_obs):
        dist, value, hxs = self.forward(state, add_state, hxs, masks, past_obs)
        if self.args.deterministic_policy:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return ActionData(value, action, action_log_probs, hxs, {
            'dist_entropy': dist_entropy
        })

    def forward(self, state, add_state, hxs, masks, past_obs):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_state)

        value = self._get_value_from_features(base_features, hxs, masks)

        if past_obs.numel() == 0:
            state = state.unsqueeze(1)
        else:
            state = torch.cat([past_obs, state.unsqueeze(1)], dim=1)

        batch_size, length, _ = state.shape

        if length > self.ctx_size:
            state = state[:, -self.ctx_size:, :]
            length = self.ctx_size

        nonzero_len = (state.abs().sum(dim=-1) != 0).sum(dim=1)
        obs_mask = self.ar_mask(batch_size, length, nonzero_len, state.device)
        with torch.no_grad():
            latent_future, _ = self.slotmae.encode(state, obs_mask)
        actor_features, _ = self.actor(base_features, latent_future.view(batch_size, 1, -1).squeeze(1), hxs, masks)
        # actor_features, _ = self.actor(base_features, hxs, masks)
        dist = self.dist(actor_features)

        return dist, value, hxs

    def evaluate_actions(self, state, add_state, hxs, masks, action, past_obs):
        #import ipdb; ipdb.set_trace()
        dist, value, hxs = self.forward(state, add_state, hxs, masks, past_obs)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        return {
            'value': value,
            'log_prob': action_log_probs,
            'ent': dist_entropy,
        }

    def get_actor_params(self):
        return super().get_actor_params() + list(self.dist.parameters())