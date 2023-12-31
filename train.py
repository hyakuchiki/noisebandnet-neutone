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
    if "raw_path" in cfg.data:
        cfg.data.raw_path = hydra.utils.to_absolute_path(cfg.data.raw_path)
    dataset = hydra.utils.instantiate(cfg.data)
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
        save_top_k=1,
        dirpath=os.path.join(os.getcwd(), "ckpts"),
        save_last=cfg.save_last,
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
        )
        callbacks.append(xfer_logger)
    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=tb_logger
    )
    tb_logger.experiment.add_text("cfg", "<pre>" + OmegaConf.to_yaml(cfg) + "</pre>")
    # train model
    trainer.fit(model, train_dl, valid_dl, ckpt_path=cfg.ckpt)


if __name__ == "__main__":
    main()
