from efficientnet_pytorch import EfficientNet
from adamp import AdamP
#from adamp import SGDP
import numpy as np
from statistics import mean
import pytorch_lightning as pl
import torch
#from accuracy import calculate_accuracy
from diffaug import DiffAugment
#from pytorch_lightning.metrics import Accuracy

class CustomTrainClass(pl.LightningModule):
  def __init__(self, model_train, num_classes, diffaug_activate, policy):
    super().__init__()
    if model_train == 'efficientnet-b0':
      self.netD = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    elif model_train == 'efficientnet-b1':
      self.netD = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    elif model_train == 'efficientnet-b2':
      self.netD = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
    elif model_train == 'efficientnet-b3':
      self.netD = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    elif model_train == 'efficientnet-b4':
      self.netD = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    elif model_train == 'efficientnet-b5':
      self.netD = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    elif model_train == 'efficientnet-b6':
      self.netD = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    elif model_train == 'efficientnet-b7':
      self.netD = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)

    #weights_init(self.netD, 'kaiming') #only use this if there is no pretrain

    self.criterion = torch.nn.CrossEntropyLoss()

    self.accuracy = []
    self.losses = []
    self.diffaug_activate = diffaug_activate
    self.accuracy_val = []
    self.losses_val = []

    self.policy = policy

  def training_step(self, train_batch, batch_idx):
    if self.diffaug_activate == False:
      preds = self.netD(train_batch[0])
    else:
      preds = self.netD(DiffAugment(train_batch[0], policy=self.policy))

    # Calculate loss
    loss = self.criterion(preds, train_batch[1])

    self.accuracy.append(calculate_accuracy(preds, train_batch[1]))
    self.losses.append(loss.item())
    return loss

  def configure_optimizers(self):
      #optimizer = torch.optim.Adam(self.netD.parameters(), lr=2e-3)
      optimizer = AdamP(self.netD.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-2)
      #optimizer = SGDP(self.netD.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
      return optimizer

  def training_epoch_end(self, training_step_outputs):
      loss_mean = np.mean(self.losses)
      #accuracy_mean = torch.mean(self.accuracy)
      accuracy_mean = torch.mean(torch.stack(self.accuracy))
      accuracy_mean = mean(self.accuracy_val)

      print(f"'Epoch': {self.current_epoch}, 'loss': {loss_mean:.2f}, 'accuracy': {accuracy_mean:.2f}")

      # logging
      self.log('train/loss_mean', loss_mean, prog_bar=True, logger=True, on_epoch=True)
      self.log('train/accuracy_mean', accuracy_mean, prog_bar=True, logger=True, on_epoch=True)

      self.losses = []
      self.accuracy = []

      torch.save(self.netD.state_dict(), f"Checkpoint_{self.current_epoch}_{self.global_step}_loss_{loss_mean:3f}_acc_{accuracy_mean:3f}_D.pth")

  def validation_step(self, train_batch, train_idx):
      preds = self.netD(train_batch[0])

      loss = self.criterion(preds, train_batch[1])
      self.accuracy_val.append(calculate_accuracy(preds, train_batch[1]).item())
      self.losses_val.append(loss.item())

  def validation_epoch_end(self, val_step_outputs):
      loss_mean = np.mean(self.losses_val)
      accuracy_mean = mean(self.accuracy_val)

      print(f"'Epoch': {self.current_epoch}, 'loss_val': {loss_mean:.2f}, 'accuracy_val': {accuracy_mean:.2f}")
      print("----------------------------------------------------------")

      # logging
      self.log('val/loss_mean', loss_mean, prog_bar=True, logger=True, on_epoch=True)
      self.log('val/accuracy_mean', accuracy_mean, prog_bar=True, logger=True, on_epoch=True)

  def test_step(self, train_batch, train_idx):
      preds = self.netD(train_batch[0])
      print("################")
      print(train_batch[1])
      print(preds.topk(k=1)[1])
