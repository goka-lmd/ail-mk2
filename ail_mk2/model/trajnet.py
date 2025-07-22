import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor, device
from typing import Union
import numpy as np
import math
from omegaconf import DictConfig

Device = Union[device, str, int, None]


class SlotMAEPE(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, slots: Tensor, observations: Tensor, goal: Tensor = None) -> Tensor:
        """
        Args:
            [batch_size, seq_len, embedding_dim]
        """

        if goal is not None:  # for encoder
            slots += self.pe[:, :slots.shape[1]]
            goal += self.pe[:, slots.shape[1]:slots.shape[1] + goal.shape[1]]
            observations += self.pe[:, slots.shape[1] + goal.shape[1]: slots.shape[1] + goal.shape[1] + observations.shape[1]]
            return slots, goal, observations
        else:  # for decoder
            slots += self.pe[:, :slots.shape[1]]
            observations += self.pe[:, slots.shape[1]: slots.shape[1] + observations.shape[1]]
            return slots, observations



class SlotMAE(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: DictConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = config.embed_dim
        self.n_slots = config.n_slots
        self.use_goal = config.use_goal

        self.positional_encoding = SlotMAEPE(d_model=self.embed_dim)
        self.slots = nn.Embedding(self.n_slots, self.embed_dim)
        # encoder
        self.obs_embed = nn.Linear(self.obs_dim, self.embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            dim_feedforward=self.embed_dim * 4,
            nhead=config.n_head,
            dropout=config.pdrop,
            activation=F.gelu,
            norm_first=True,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, config.n_enc_layers)
        self.encoder_norm = nn.LayerNorm(self.embed_dim)

        # decoder

        self.obs_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.obs_decode_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            dim_feedforward=self.embed_dim * 4,
            nhead=config.n_head,
            dropout=config.pdrop,
            activation=F.gelu,
            norm_first=True,
            batch_first=True
        )

        self.decoder = nn.TransformerEncoder(self.decoder_layer, config.n_dec_layers)

        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.obs_decoder = nn.Linear(self.embed_dim, self.obs_dim)
    
    def encode(self, observations: Tensor, obs_mask: Tensor):
        batch_size = observations.shape[0]
        slots = self.slots(torch.arange(self.n_slots, device=observations.device)).repeat(batch_size, 1, 1)

        # obs: B X L x obs_dim
        obs_embeddings = self.obs_embed(observations)

        s, o = self.positional_encoding(slots, obs_embeddings)

        o_keep_list = []
        for i in range(batch_size):
            valid_o = o[i][obs_mask[i] == 0]
            o_keep_list.append(valid_o)

        from torch.nn.utils.rnn import pad_sequence
        o_keep = pad_sequence(o_keep_list, batch_first=True)
        # o_keep = o[obs_mask == 0].view(batch_size, -1, self.embed_dim)
        
        enc_inputs = torch.cat([s, o_keep], dim=1)
    
        encoded_keep = self.encoder(enc_inputs)
        encoded_keep = self.encoder_norm(encoded_keep)

        bottleneck = encoded_keep[:, :s.shape[1]]
        encoded_obs = encoded_keep[:, s.shape[1] :]

        return bottleneck, encoded_obs
    
    def decode(self, bottleneck: Tensor, obs_mask: Tensor):
        batch_size = bottleneck.shape[0]
        mask_obs = self.obs_mask_token.repeat(obs_mask.shape[0], obs_mask.shape[1], 1)

        b, o = self.positional_encoding(bottleneck, mask_obs)

        dec_inputs = torch.cat([b, o], dim=1)
        decode_out = self.decoder(dec_inputs)
        decode_out = self.decoder_norm(decode_out)

        obs_out = decode_out[:, b.shape[1]: ]

        pred_o = self.obs_decoder(obs_out)

        return pred_o

    def forward(self, observations: Tensor, obs_mask: Tensor):
        '''
        obs_mask: boolean tensor, True means masked
        '''
        bottleneck, _ = self.encode(observations, obs_mask)
        pred_o = self.decode(bottleneck, obs_mask)

        return pred_o, bottleneck


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 0: fake, 1: expert
        )

    def forward(self, traj_repr):
        return self.model(traj_repr)
    

class TrajNet(pl.LightningModule):
    def __init__(self, seed: int, env_name: str, obs_dim: int, action_dim: int, lr: float, epochs: int, ctx_size: int, future_horizon: int, stage: str, model_config: DictConfig, **kwargs):
        super().__init__()

        self.env_name = env_name
        self.ctx_size = ctx_size
        self.future_horizon = future_horizon
        self.stage = stage
        self.mask_type = model_config.mask_type
        self.ar_mask_ratios = model_config.ar_mask_ratios
        self.rnd_mask_ratios = model_config.rnd_mask_ratios
        self.ar_mask_ratio_weights = model_config.ar_mask_ratio_weights
        self.traj_len = self.ctx_size + self.future_horizon
        self.lr = lr
        self.num_epochs = epochs
        self.model = SlotMAE(obs_dim, action_dim, model_config)

        self.discriminator = TrajectoryDiscriminator(input_dim=model_config.embed_dim * model_config.n_slots)
        self.adv_weight = model_config.get("adv_weight", 0.01)
        
        self.save_hyperparameters()
    
    def forward(self, observations: Tensor, obs_mask: Tensor):
        return self.model.forward(observations, obs_mask)

    def loss_gen(self, target_o: Tensor, pred_o: Tensor, padding_mask: Tensor, obs_mask: Tensor, slot_out: Tensor):
        # masked autoencoder loss
        B, T = padding_mask.size()
        padding_mask = padding_mask.float()
        
        valid_mask = (obs_mask.bool() & (padding_mask == 0))
        valid_mask = valid_mask.unsqueeze(-1).expand_as(pred_o)

        masked_pred = pred_o[valid_mask]
        masked_target = target_o[valid_mask]

        if masked_pred.numel() > 0:
            loss_o = F.mse_loss(masked_pred, masked_target, reduction='mean')
        else:
            loss_o = torch.tensor(0.0, device=pred_o.device)

        # discriminator loss
        slot_out_flat = slot_out.reshape(B, -1)
        pred_disc_fake = self.discriminator(slot_out_flat)
        loss_adv = -torch.log(pred_disc_fake + 1e-8).mean()

        return loss_o + self.adv_weight * loss_adv

    def loss_disc(self, slot_out_fake, slot_out_real):
        B = slot_out_fake.size(0)

        fake_flat = slot_out_fake.reshape(B, -1).detach()
        real_flat = slot_out_real.reshape(B, -1).detach()

        pred_fake = self.discriminator(fake_flat)
        pred_real = self.discriminator(real_flat)

        loss_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
        loss_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))

        return loss_fake + loss_real
    
    def ar_mask(self, batch_size: int, length: int, keep_len: float, device: Device):
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        return mask

    def rnd_mask(self, batch_size: int, length: int, mask_ratio: float, device: Device):
        keep_len = max(1, int(length * (1 - mask_ratio)))  # at least keep the first obs

        noise = torch.rand(size=(batch_size, length), device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones([batch_size, length], device=device)
        mask[:, :keep_len] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def training_step(self, batch, batch_idx, optimizer_idx):
        observations, valid_length, padding_mask = batch
        batch_size, _, _ = observations.shape
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]

        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        if self.mask_type == 'mae_all':
            keep_len = 1
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            obs_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'ae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.ar_mask(batch_size, keep_len, keep_len, observations.device)
        elif self.mask_type == 'mae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'mae_f':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            obs_mask = self.ar_mask(batch_size, obs_length, keep_len, observations.device)
        elif self.mask_type == 'mae_rc':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))
            history_rnd_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            future_causal_mask = self.ar_mask(batch_size, obs_length - keep_len, 0, observations.device)
            obs_mask = torch.cat([history_rnd_mask, future_causal_mask], dim=1)
        else:
            raise NotImplementedError

        # --------- optimizer_idx = 0 → Update generator ---------
        if optimizer_idx == 0:
            pred_o, slot_out = self(observations, obs_mask)
            loss = self.loss_gen(observations, pred_o, padding_mask, obs_mask, slot_out)

            self.log("train/loss_gen", loss, sync_dist=True)
            return loss

        # --------- optimizer_idx = 1 → Update discriminator ---------
        elif optimizer_idx == 1:
            with torch.no_grad():
                pred_o, _ = self(observations, obs_mask)
                _, slot_out_fake = self(pred_o, torch.zeros_like(obs_mask))
                _, slot_out_real = self(observations, torch.zeros_like(obs_mask))

            loss = self.loss_disc(slot_out_fake, slot_out_real)
            self.log("train/loss_disc", loss, sync_dist=True)
            return loss        

    def validation_step(self, batch, batch_idx):
        observations, valid_length, padding_mask = batch
        batch_size, _, _ = observations.shape
        ar_mask_ratio = np.random.choice(self.ar_mask_ratios, 1, p=self.ar_mask_ratio_weights)[0]
        rnd_mask_ratio = np.random.choice(self.rnd_mask_ratios, 1)[0]

        # For convenience, make sure the number of unmasked (obs/action) is the same across examples when masking
        if self.mask_type == 'mae_all':
            keep_len = 1
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            obs_mask = self.rnd_mask(batch_size, obs_length, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'ae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.ar_mask(batch_size, keep_len, keep_len, observations.device)
        elif self.mask_type == 'mae_h':
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            observations = observations[:, :keep_len]
            padding_mask = padding_mask[:, :keep_len]
            obs_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
        elif self.mask_type == 'mae_f':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))  # match the beginning of eval (obs length is less than ctx size)
            obs_mask = self.ar_mask(batch_size, obs_length, keep_len, observations.device)
        elif self.mask_type == 'mae_rc':
            observations = observations[:, :self.traj_len]
            padding_mask = padding_mask[:, :self.traj_len]
            _, obs_length, _ = observations.shape
            keep_len = max(1, int(self.ctx_size * (1 - ar_mask_ratio)))
            history_rnd_mask = self.rnd_mask(batch_size, keep_len, rnd_mask_ratio, observations.device)
            future_causal_mask = self.ar_mask(batch_size, obs_length - keep_len, 0, observations.device)
            obs_mask = torch.cat([history_rnd_mask, future_causal_mask], dim=1)
        else:
            raise NotImplementedError
        
        pred_o, slot_out = self(observations, obs_mask)
        loss = self.loss_gen(observations, pred_o, padding_mask, obs_mask, slot_out)

        self.log_dict({
            'val/val_loss': loss,
            'val/disc_output_mean': self.discriminator(slot_out.reshape(batch_size, -1)).mean().item()
            },  
        sync_dist=True)
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr * 0.5)
        return [opt_g, opt_d], []

