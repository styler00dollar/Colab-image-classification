import argparse
from dataloader import DataModule
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl

import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


def main():
    dm = DataModule(training_path=cfg['training_path'], validation_path=cfg['validation_path'], test_path=cfg['test_path'], num_workers = cfg['num_workers'], size = cfg['size'], batch_size=cfg['batch_size'], means=cfg['means'], std=cfg['std'])
    model = CustomTrainClass(model_train=cfg['model_train'], num_classes=cfg['num_classes'], diffaug_activate=cfg['diffaug_activate'], policy=cfg['policy'], aug=cfg['aug'], timm=cfg['timm'])

    # skipping validation with limit_val_batches=0
    #gpus=1, limit_val_batches=0,
    trainer = pl.Trainer(gpus=1, max_epochs=800, progress_bar_refresh_rate=20, default_root_dir=cfg['default_root_dir'])

    # For resuming training
    """
    checkpoint_path = None #'test.ckpt'

    if checkpoint_path is not None:
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, gpus=1, max_epochs=800, progress_bar_refresh_rate=20, default_root_dir=args.default_root_dir)

        model = model.load_from_checkpoint(checkpoint_path)
        dm = DataModule(batch_size=16, training_path=training_path, validation_path=args.validation_path, num_workers = 16, size = 256)
        checkpoint = torch.load(checkpoint_path)
        trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
        trainer.checkpoint_connector.restore_training_state(checkpoint)
        pl.Trainer.global_step = checkpoint['global_step']
        pl.Trainer.epoch = checkpoint['epoch']
        print("Checkpoint was loaded successfully.")

        #############################################
    """
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
