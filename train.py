from dataloader import DataModule
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl
from checkpoint import CheckpointEveryNSteps

import yaml
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

def main():
    dm = DataModule(training_path=cfg['path']['training_path'], validation_path=cfg['path']['validation_path'], test_path=cfg['path']['test_path'], num_workers = cfg['num_workers'], size = cfg['size'], batch_size=cfg['batch_size'], means=cfg['means'], std=cfg['std'])
    model = CustomTrainClass(model_train=cfg['model_train'], num_classes=cfg['num_classes'], diffaug_activate=cfg['diffaug_activate'], policy=cfg['policy'], aug=cfg['aug'], timm=cfg['timm'])

    # skipping validation with limit_val_batches=0
    if cfg['use_amp'] == False:
      trainer = pl.Trainer(num_sanity_val_steps=0, stochastic_weight_avg=cfg['use_swa'], log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, gpus=cfg['gpus'], max_epochs=['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])
    if cfg['use_amp'] == True:
      trainer = pl.Trainer(num_sanity_val_steps=0, stochastic_weight_avg=cfg['use_swa'], log_every_n_steps=50, resume_from_checkpoint=cfg['path']['checkpoint_path'], check_val_every_n_epoch=9999999, logger=None, gpus=cfg['gpus'], precision=16, amp_level='O1', max_epochs=cfg['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['save_step_frequency'], save_path=cfg['path']['checkpoint_save_path'])])

    #############################################
    # Loading a Model
    #############################################
    # For resuming training
    if cfg['path']['checkpoint_path'] is not None:
      # load from checkpoint (optional) (using a model as pretrain and disregarding other parameters)
      #model = model.load_from_checkpoint(checkpoint_path) # start training from checkpoint, warning: apperantly global_step will be reset to zero and overwriting validation images, you could manually make an offset

      # continue training with checkpoint (does restore values) (optional)
      # https://github.com/PyTorchLightning/pytorch-lightning/issues/2613
      # https://pytorch-lightning.readthedocs.io/en/0.6.0/pytorch_lightning.trainer.training_io.html
      # https://github.com/PyTorchLightning/pytorch-lightning/issues/4333
      # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])

      # To use DDP for local multi-GPU training, you need to add find_unused_parameters=True inside the DDP command
      model = model.load_from_checkpoint(cfg['path']['checkpoint_path'])
      #trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, logger=None, gpus=cfg['gpus'], max_epochs=cfg['datasets']['train']['max_epochs'], progress_bar_refresh_rate=cfg['progress_bar_refresh_rate'], default_root_dir=cfg['default_root_dir'], callbacks=[CheckpointEveryNSteps(save_step_frequency=cfg['datasets']['train']['save_step_frequency'], save_path = cfg['path']['checkpoint_save_path'])])
      checkpoint = torch.load(cfg['path']['checkpoint_path'])
      trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
      trainer.checkpoint_connector.restore_training_state(checkpoint)
      pl.Trainer.global_step = checkpoint['global_step']
      pl.Trainer.epoch = checkpoint['epoch']
      print("Checkpoint was loaded successfully.")

    #############################################

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()