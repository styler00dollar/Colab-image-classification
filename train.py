from dataloader import DataModule
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl

import yaml

import torch

torch.set_float32_matmul_precision("medium")

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def main():
    if cfg["ffcv"]:
        import torch
        from ffcv.fields.basics import IntDecoder
        from ffcv.loader import OrderOption
        from ffcv.transforms import (
            ToTensor,
            ToTorchImage,
            NormalizeImage,
        )
        from ffcv_pl.data_loading import FFCVDataModule
        from ffcv_pl.ffcv_utils.augmentations import DivideImage255
        from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager
        from torchvision.transforms import RandomHorizontalFlip
        import numpy as np

        ffcv_pipeline = FFCVPipelineManager(
            cfg["path"]["beton_path"],  # dataset_creation.py
            pipeline_transforms=[
                # image pipeline
                # todo: augmentation
                [
                    NormalizeImage(
                        mean=np.array(cfg["means"]),
                        std=np.array(cfg["std"]),
                        type=np.float32,
                    ),
                    ToTensor(),
                    ToTorchImage(),
                    DivideImage255(dtype=torch.float32),
                    RandomHorizontalFlip(p=0.5),
                ],
                # label (int) pipeline
                [IntDecoder(), ToTensor()],
            ],
            ordering=OrderOption.RANDOM,
        )

        # todo: validation
        dm = FFCVDataModule(
            batch_size=cfg["batch_size"],
            is_dist=False,
            num_workers=cfg["num_workers"],
            train_manager=ffcv_pipeline,
            val_manager=ffcv_pipeline,
        )
    if not cfg["ffcv"]:
        dm = DataModule(
            training_path=cfg["path"]["training_path"],
            validation_path=cfg["path"]["validation_path"],
            test_path=cfg["path"]["test_path"],
            num_workers=cfg["num_workers"],
            size=cfg["size"],
            batch_size=cfg["batch_size"],
            means=cfg["means"],
            std=cfg["std"],
        )

    model = CustomTrainClass(
        model_train=cfg["model_train"],
        num_classes=cfg["num_classes"],
        diffaug_activate=cfg["diffaug_activate"],
        policy=cfg["policy"],
    )

    callbacks = []

    if cfg["use_swa"]:
        print("Using SWA")
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2))

    if cfg["lr_finder"]:
        print("Looking for LR")
        callbacks.append(pl.callbacks.LearningRateFinder())

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        callbacks=callbacks,
        accelerator="gpu",
        log_every_n_steps=50,
        check_val_every_n_epoch=None,
        logger=None,
        precision=cfg["precision"],
        max_epochs=cfg["max_epochs"],
        default_root_dir=cfg["default_root_dir"],
        val_check_interval=cfg["save_step_frequency"],
    )

    if cfg["path"]["pretrain"]:
        import torch

        model.netD.load_state_dict(torch.load(cfg["path"]["pretrain"]), strict=False)
        print("Pretrain pth loaded!")

    #############################################
    # Loading a Model
    #############################################
    # For resuming training
    if cfg["path"]["checkpoint_path"] is not None:
        # load from checkpoint (optional) (using a model as pretrain and disregarding other parameters)
        # model = model.load_from_checkpoint(checkpoint_path) # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset

        # continue training with checkpoint (does restore values) (optional)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2613
        # https://pytorch-lightning.readthedocs.io/en/0.6.0/pytorch_lightning.trainer.training_io.html
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4333
        # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])

        # To use DDP for local multi-GPU training, you need to add find_unused_parameters=True inside the DDP command
        model = model.load_from_checkpoint(cfg["path"]["checkpoint_path"])
        # trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, logger=None, gpus=cfg['gpus'], max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path = cfg['path']['checkpoint_save_path'])])
        checkpoint = torch.load(cfg["path"]["checkpoint_path"])
        trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
        trainer.checkpoint_connector.restore_training_state(checkpoint)
        pl.Trainer.global_step = checkpoint["global_step"]
        pl.Trainer.epoch = checkpoint["epoch"]
        print("Checkpoint was loaded successfully.")

    #############################################

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
