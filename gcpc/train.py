import os
import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from .data.sequence import MAEDataModule

import hydra
from omegaconf import DictConfig


def make_ckpt_filename(cfg: DictConfig):
    ckptfile_name = f'{cfg.model._target_.split(".")[2]}-opts:{cfg.opts}'
    return ckptfile_name


@hydra.main(version_base=None, config_path='./config', config_name='train')
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    assert cfg.traj_load_path is not None, "Must specify expert demonstrations!"

    dm = MAEDataModule(cfg)
    if hasattr(cfg.model, 'stage'):
        dm.setup(stage=cfg.model.stage)
    else:
        dm.setup('fit')


    gcpc = hydra.utils.instantiate(cfg.model,
        seed=cfg.seed,
        env_name=cfg.env_name,
        obs_dim=dm.get_obs_dim(),
        action_dim=dm.get_action_dim(),
        epochs=cfg.exp.epochs,
        lr=cfg.exp.lr,
        use_scheduler=cfg.exp.use_scheduler,
        ctx_size=cfg.exp.ctx_size,
        future_horizon=cfg.exp.future_horizon,
        num_eval_rollouts=cfg.exp.num_eval_rollouts
    )

    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    wdb_logger = WandbLogger(name=f'{cfg.env_name}-{make_ckpt_filename(cfg)}', save_dir=working_dir, project=cfg.wandb.project, log_model=True)

    progressbar = TQDMProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(working_dir, 'lightning_logs', cfg.env_name, make_ckpt_filename(cfg), 'checkpoints'),
        monitor='epoch',
        mode='max',
        save_last=True,
        save_top_k=-1,
    )

    if hasattr(cfg.model, 'stage') and cfg.model.stage == 'trl':
        checkpoint = ModelCheckpoint(
            monitor='val/val_loss',
            dirpath=os.path.join(working_dir, 'lightning_logs', cfg.env_name, make_ckpt_filename(cfg), 'checkpoints'),
            filename=cfg.env_name + '-epoch:{epoch}-val_loss:{val/val_loss:.3f}',
            mode='min',
            save_last=True,
            save_top_k=-1,
            auto_insert_metric_name=False
        )


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.devices,
        max_epochs=cfg.exp.epochs,
        logger=wdb_logger,
        gradient_clip_val=1.0,
        callbacks=[progressbar, lr_monitor, checkpoint],
    )

    print(f'\nSeed: {cfg.seed}')
    print(f'CUDA devices: {cfg.devices}')
    print(f'Log Directory: {os.path.join(cfg.env_name, make_ckpt_filename(cfg))}\n')

    trainer.fit(model=gcpc, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    main()
