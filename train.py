import hydra


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.1")
def main(cfg):
    import os, warnings, shutil
    import pytorch_lightning as pl
    import torch
    from datetime import timedelta
    from omegaconf import OmegaConf
    from ddsp.log import AudioLogger, XferLogger, CkptEveryNSteps
    from torch.utils.data import DataLoader, random_split
    from pytorch_lightning.callbacks import ModelCheckpoint
    from ddsp.model import AutoEncoderModel

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    warnings.simplefilter("once")
    # load model
    model = AutoEncoderModel(cfg.ae, cfg.loss)
    # loggers setup
    tb_logger = pl.loggers.TensorBoardLogger(
        "tb_logs", "", default_hp_metric=False, version=""
    )
    # load dataset
    print("Starting Preprocessing.")
    if "db_path" in cfg.data:
        cfg.data.db_path = hydra.utils.to_absolute_path(cfg.data.db_path)
    if "raw_path" in cfg.data:
        cfg.data.raw_path = hydra.utils.to_absolute_path(cfg.data.raw_path)
    if "raw_dir" in cfg.data:
        cfg.data.raw_dir = hydra.utils.to_absolute_path(cfg.data.raw_dir)
    dataset = hydra.utils.instantiate(cfg.data)
    print(f"Loaded {len(dataset)} samples.")
    train_set, valid_set = random_split(
        dataset,
        [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_dl = DataLoader(
        train_set,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    # trainer setup
    # keep every checkpoint_every epochs and best epoch
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        dirpath=os.path.join(os.getcwd(), "ckpts"),
        monitor=cfg.monitor,
        save_last=cfg.save_last,
        train_time_interval=timedelta(
            minutes=cfg.save_every_n_min
        ),  # save ckpt every hour
        save_on_train_epoch_end=True,
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        AudioLogger(sr=cfg.sample_rate),
        checkpoint_callback,
    ]
    if cfg.xfer_dir is not None:
        xfer_logger = XferLogger(
            hydra.utils.to_absolute_path(cfg.xfer_dir),
            sr=cfg.sample_rate,
            feat_proc=dataset.feat_proc,
            fad=cfg.get("monitor_fad", False),
        )
        callbacks.append(xfer_logger)
    if cfg.save_every_n_steps is not None:
        callbacks.append(CkptEveryNSteps(cfg.save_every_n_steps, len(train_dl)))
    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=tb_logger
    )
    tb_logger.experiment.add_text("cfg", "<pre>" + OmegaConf.to_yaml(cfg) + "</pre>")
    # train model
    trainer.fit(model, train_dl, valid_dl, ckpt_path=cfg.ckpt)
    # torch.save(datamodule, os.path.join(os.getcwd(), 'datamodule.pt'))
    # log some specified hparams
    if cfg.monitor:
        tb_logger.log_hyperparams(
            {hp_name: cfg.get(hp_name) for hp_name in cfg.hparams},
            metrics=trainer.callback_metrics[cfg.monitor],
        )
        # return value used for optuna
        return trainer.callback_metrics[cfg.monitor]


if __name__ == "__main__":
    main()
