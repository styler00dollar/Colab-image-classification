from efficientnet_pytorch import EfficientNet
from adamp import AdamP
#from adamp import SGDP
import numpy as np
from statistics import mean
import pytorch_lightning as pl
import torch
from accuracy import calculate_accuracy, calc_accuracy_gridmix
from diffaug import DiffAugment
import yaml
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

from tensorboardX import SummaryWriter
writer = SummaryWriter(logdir=cfg['path']['log_path'])

if cfg['aug'] == 'MuAugment':
  from MuarAugment import BatchRandAugment, MuAugment

class CustomTrainClass(pl.LightningModule):
  def __init__(self, model_train='tf_efficientnetv2_b0', num_classes=3, diffaug_activate=False, policy='color,translation', aug=None):
    super().__init__()


    #############################################
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



    elif model_train == 'mobilenetv3_small':
      from arch.mobilenetv3_arch import MobileNetV3
      self.netD = MobileNetV3(n_class=num_classes, mode='small', input_size=256)
    elif model_train == 'mobilenetv3_large':
      from arch.mobilenetv3_arch import MobileNetV3
      self.netD = MobileNetV3(n_class=num_classes, mode='large', input_size=256)



    elif model_train == 'resnet50':
      from arch.resnet_arch import resnet50
      self.netD = resnet50(num_classes=num_classes, pretrain=cfg['pretrain'])
    elif model_train == 'resnet101':
      from arch.resnet_arch import resnet101
      self.netD = resnet101(num_classes=num_classes, pretrain=cfg['pretrain'])
    elif model_train == 'resnet152':
      from arch.resnet_arch import resnet152
      self.netD = resnet152(num_classes=num_classes, pretrain=cfg['pretrain'])

    #############################################
    elif model_train == 'ViT':
      from vit_pytorch import ViT
      self.netD = ViT(
          image_size = 256,
          patch_size = 32,
          num_classes = num_classes,
          dim = 1024,
          depth = 6,
          heads = 16,
          mlp_dim = 2048,
          dropout = 0.1,
          emb_dropout = 0.1
      )

    elif model_train == 'DeepViT':
      from vit_pytorch.deepvit import DeepViT
      self.netD = DeepViT(
          image_size = 256,
          patch_size = 32,
          num_classes = num_classes,
          dim = 1024,
          depth = 6,
          heads = 16,
          mlp_dim = 2048,
          dropout = 0.1,
          emb_dropout = 0.1
      )


    #############################################

    elif model_train == 'RepVGG-A0':
      from arch.RepVGG_arch import create_RepVGG_A0
      self.netD = create_RepVGG_A0(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-A1':
      from arch.RepVGG_arch import create_RepVGG_A1
      self.netD = create_RepVGG_A1(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-A2':
      from arch.RepVGG_arch import create_RepVGG_A2
      self.netD = create_RepVGG_A2(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B0':
      from arch.RepVGG_arch import create_RepVGG_B0
      self.netD = create_RepVGG_B0(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B1':
      from arch.RepVGG_arch import create_RepVGG_B1
      self.netD = create_RepVGG_B1(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B1g2':
      from arch.RepVGG_arch import create_RepVGG_B1g2
      self.netD = create_RepVGG_B1g2(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B1g4':
      from arch.RepVGG_arch import create_RepVGG_B1g4
      self.netD = create_RepVGG_B1g4(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B2':
      from arch.RepVGG_arch import create_RepVGG_B2
      self.netD = create_RepVGG_B2(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B2g2':
      from arch.RepVGG_arch import create_RepVGG_B2g2
      self.netD = create_RepVGG_B2g2(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B2g4':
      from arch.RepVGG_arch import create_RepVGG_B2g4
      self.netD = create_RepVGG_B2g4(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B3':
      from arch.RepVGG_arch import create_RepVGG_B3
      self.netD = create_RepVGG_B3(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B3g2':
      from arch.RepVGG_arch import create_RepVGG_B3g2
      self.netD = create_RepVGG_B3g2(deploy=False, num_classes=num_classes)

    elif model_train == 'RepVGG-B3g4':
      from arch.RepVGG_arch import create_RepVGG_B3g4
      self.netD = create_RepVGG_B3g4(deploy=False, num_classes=num_classes)

    #############################################

    elif model_train == 'squeezenet_1_0':
      from arch.squeezenet_arch import SqueezeNet
      self.netD = SqueezeNet(num_classes=num_classes, version='1_0')

    elif model_train == 'squeezenet_1_1':
      from arch.squeezenet_arch import SqueezeNet
      self.netD = SqueezeNet(num_classes=num_classes, version='1_1')
    #############################################
    elif model_train == 'vgg11':
      from arch.vgg_arch import create_vgg11
      self.netD = create_vgg11(num_classes, pretrained=cfg['pretrain'])
    elif model_train == 'vgg13':
      from arch.vgg_arch import create_vgg13
      self.netD = create_vgg13(num_classes, pretrained=cfg['pretrain'])
    elif model_train == 'vgg16':
      from arch.vgg_arch import create_vgg16
      self.netD = create_vgg16(num_classes, pretrained=cfg['pretrain'])
    elif model_train == 'vgg19':
      from arch.vgg_arch import create_vgg19
      self.netD = create_vgg19(num_classes, pretrained=cfg['pretrain'])

    #############################################
    elif model_train == 'SwinTransformer':
      from swin_transformer_pytorch import SwinTransformer

      self.netD = SwinTransformer(
          hidden_dim=96,
          layers=(2, 2, 6, 2),
          heads=(3, 6, 12, 24),
          channels=3,
          num_classes=num_classes,
          head_dim=32,
          window_size=8,
          downscaling_factors=(4, 2, 2, 2),
          relative_pos_embedding=True
      )

    elif model_train == 'effV2':
      if cfg['size'] == "s":
        from arch.efficientnetV2_arch import effnetv2_s
        self.netD = effnetv2_s(num_classes=num_classes)
      elif cfg['size'] == "m":
        from arch.efficientnetV2_arch import effnetv2_m
        self.netD = effnetv2_m(num_classes=num_classes)
      elif cfg['size'] == "l":
        from arch.efficientnetV2_arch import effnetv2_l
        self.netD = effnetv2_l(num_classes=num_classes)
      elif cfg['size'] == "xl":
        from arch.efficientnetV2_arch import effnetv2_xl
        self.netD = effnetv2_xl(num_classes=num_classes)



    elif model_train == 'x_transformers':
      from x_transformers import ViTransformerWrapper, Encoder
      self.netD = ViTransformerWrapper(
          image_size = cfg['image_size'],
          patch_size = cfg['patch_size'],
          num_classes = num_classes,
          attn_layers = Encoder(
              dim = cfg['dim'],
              depth = cfg['depth'],
              heads = cfg['heads'],
          )
      )
        
    elif model_train == 'mobilevit':
      if cfg['model_size'] == "xxs":
        from arch.mobilevit_arch import mobilevit_xxs
        self.netD = mobilevit_xxs(num_classes=num_classes)
      elif cfg['model_size'] == "xs":
        from arch.mobilevit_arch import mobilevit_xs
        self.netD = mobilevit_xs(num_classes=num_classes)
      elif cfg['model_size'] == "x":
        from arch.mobilevit_arch import mobilevit_s
        self.netD = mobilevit_s(num_classes=num_classes)

    elif model_train == 'hrt':
      from arch.hrt_arch import HighResolutionTransformer
      self.netD = HighResolutionTransformer(num_classes)

    elif model_train == 'volo':
      if cfg['model_size'] == "volo_d1":
        from arch.volo_arch import volo_d1
        self.netD = volo_d1(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "volo_d2":
        from arch.volo_arch import volo_d2
        self.netD = volo_d2(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "volo_d3":
        from arch.volo_arch import volo_d3
        self.netD = volo_d3(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "volo_d4":
        from arch.volo_arch import volo_d4
        self.netD = volo_d4(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "volo_d5":
        from arch.volo_arch import volo_d5
        self.netD = volo_d5(pretrained=cfg['pretrain'], num_classes=num_classes)

    elif model_train == 'pvt_v2':
      if cfg['model_size'] == "pvt_v2_b0":
        from arch.pvt_v2_arch import pvt_v2_b0
        self.netD = pvt_v2_b0(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "pvt_v2_b1":
        from arch.pvt_v2_arch import pvt_v2_b1
        self.netD = pvt_v2_b1(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "pvt_v2_b2":
        from arch.pvt_v2_arch import pvt_v2_b2
        self.netD = pvt_v2_b2(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "pvt_v2_b3":
        from arch.pvt_v2_arch import pvt_v2_b3
        self.netD = pvt_v2_b3(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "pvt_v2_b4":
        from arch.pvt_v2_arch import pvt_v2_b4
        self.netD = pvt_v2_b4(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "pvt_v2_b5":
        from arch.pvt_v2_arch import pvt_v2_b5
        self.netD = pvt_v2_b5(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "pvt_v2_b2_li":
        from arch.pvt_v2_arch import pvt_v2_b2_li
        self.netD = pvt_v2_b2_li(pretrained=cfg['pretrain'], num_classes=num_classes)

    elif model_train == 'ConvMLP':
      if cfg['model_size'] == "convmlp_s":
        from arch.ConvMLP_arch import convmlp_s
        self.netD = convmlp_s(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "convmlp_m":
        from arch.ConvMLP_arch import convmlp_m
        self.netD = convmlp_m(pretrained=cfg['pretrain'], num_classes=num_classes)
      elif cfg['model_size'] == "convmlp_l":
        from arch.ConvMLP_arch import convmlp_l
        self.netD = convmlp_l(pretrained=cfg['pretrain'], num_classes=num_classes)

    elif model_train == 'FocalTransformer':
      from arch.focal_transformer_arch import FocalTransformer
      self.netD = FocalTransformer(num_classes=num_classes)

    elif model_train == 'mobile_former':
      from arch.mobile_former_arch import MobileFormer, config_52, config_294, config_508
      if cfg['model_size'] == "config_52":
        self.netD = MobileFormer(config_52)
      elif cfg['model_size'] == "config_294":
        self.netD = MobileFormer(config_294)
      elif cfg['model_size'] == "config_508":
        self.netD = MobileFormer(config_508)

    elif model_train == 'timm':
      import timm
      self.netD = timm.create_model(cfg['model_choise'], num_classes=num_classes, pretrained=True)

    #weights_init(self.netD, 'kaiming') #only use this if there is no pretrain


    if aug == 'gridmix':
      from GridMixupLoss import GridMixupLoss
      self.criterion = GridMixupLoss(
        alpha=(0.4, 0.7),
        hole_aspect_ratio=1.,
        crop_area_ratio=(0.5, 1),
        crop_aspect_ratio=(0.5, 2),
        n_holes_x=(2, 6)
      )
    elif aug == 'cutmix':
      from cutmix import cutmix
      self.criterion = cutmix(
        alpha=(0.4, 0.7),
        hole_aspect_ratio=1.,
        crop_area_ratio=(0.5, 1),
        crop_aspect_ratio=(0.5, 2),
        n_holes_x=(2, 6)
      )

    self.aug = aug

    if cfg['loss'] == 'CenterLoss':
      from centerloss import CenterLoss
      self.criterion = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
    elif cfg['loss'] == 'normal':
      self.criterion = torch.nn.CrossEntropyLoss()

    self.accuracy = []
    self.losses = []
    self.diffaug_activate = diffaug_activate
    self.accuracy_val = []
    self.losses_val = []

    self.policy = policy
    self.iter_check = 0


    if cfg['aug'] == 'MuAugment':
      rand_augment = BatchRandAugment(N_TFMS=3, MAGN=3, mean=cfg['means'], std=cfg['std'])
      self.mu_transform = MuAugment(rand_augment, N_COMPS=4, N_SELECTED=2)

  def forward(self, x):
    return self.netD(x)

  def training_step(self, train_batch, batch_idx):
    if self.trainer.global_step != 0:
      if self.iter_check == self.trainer.global_step:
        self.trainer.global_step += 1
      self.iter_check = self.trainer.global_step


    if self.aug == 'gridmix' or self.aug == 'cutmix':
      train_batch[0], train_batch[1] = self.criterion.get_sample(images=train_batch[0], targets=train_batch[1].unsqueeze(-1))
    elif self.aug == 'MuAugment':
      self.mu_transform.setup(self)
      train_batch[0], train_batch[1] = self.mu_transform((train_batch[0], train_batch[1]))

    #elif self.aug == 'cutmix':
    #  train_batch[0], train_batch[1] = cutmix(train_batch[0], train_batch[1].unsqueeze(-1), 1)
    #print(train_batch[0].shape)

    if self.diffaug_activate == False:
      preds = self.netD(train_batch[0])
    else:
      preds = self.netD(DiffAugment(train_batch[0], policy=self.policy))

    # Calculate loss
    print(preds.shape, train_batch[1].shape)
    loss = self.criterion(preds, train_batch[1])
    writer.add_scalar('loss', loss, self.trainer.global_step)

    if cfg['print_training_epoch_end_metrics'] == False:
      if self.aug == None or self.aug == 'centerloss' or self.aug == 'MuAugment':
        acc = calculate_accuracy(preds, train_batch[1])
      else:
        acc = calc_accuracy_gridmix(preds, train_batch[1])
      writer.add_scalar('acc', acc, self.trainer.global_step)
      self.accuracy.append(acc)
      self.losses.append(loss.item())

    return loss

  def configure_optimizers(self):
      if cfg['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(self.netD.parameters(), lr=cfg['lr'])
      elif cfg['optimizer'] == "AdamP":
        optimizer = AdamP(self.netD.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=1e-2)
      elif cfg['optimizer'] == "SGDP":
        optimizer = SGDP(self.netD.parameters(), lr=cfg['lr'], weight_decay=1e-5, momentum=0.9, nesterov=True)
      elif cfg['optimizer'] == "MADGRAD":
        from madgrad import MADGRAD
        optimizer = MADGRAD(self.netD.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.01, eps=1e-6)
      return optimizer

  def training_epoch_end(self, training_step_outputs):
      if cfg['print_training_epoch_end_metrics'] == False:
        loss_mean = np.mean(self.losses)
        #accuracy_mean = torch.mean(self.accuracy)

        if self.aug == None or self.aug == 'centerloss':
          accuracy_mean = torch.mean(torch.stack(self.accuracy))
        else:
          accuracy_mean = np.mean(self.accuracy)

        print(f"'Epoch': {self.current_epoch}, 'loss': {loss_mean:.2f}, 'accuracy': {accuracy_mean:.2f}")

        # logging
        #self.log('train/loss_mean', loss_mean, prog_bar=True, logger=True, on_epoch=True)
        #self.log('train/accuracy_mean', accuracy_mean, prog_bar=True, logger=True, on_epoch=True)

        self.losses = []
        self.accuracy = []

      torch.save(self.netD.state_dict(), f"Checkpoint_{self.current_epoch}_{self.global_step}_loss_{loss_mean:3f}_acc_{accuracy_mean:3f}_D.pth")

  def validation_step(self, train_batch, train_idx):
      #print(train_batch[0].shape)
      preds = self.netD(train_batch[0])

      if self.aug == None or self.aug == 'centerloss':
        loss = self.criterion(preds, train_batch[1])
        self.losses_val.append(loss.item())

      self.accuracy_val.append(calculate_accuracy(preds, train_batch[1]).item())

  def validation_epoch_end(self, val_step_outputs):
      loss_mean = np.mean(self.losses_val)
      accuracy_mean = mean(self.accuracy_val)

      #print(f"'Epoch': {self.current_epoch}, 'loss_val': {loss_mean:.2f}, 'accuracy_val': {accuracy_mean:.2f}")
      #print("----------------------------------------------------------")

      # logging
      #self.log('val/loss_mean', loss_mean, prog_bar=True, logger=True, on_epoch=True)
      #self.log('val/accuracy_mean', accuracy_mean, prog_bar=True, logger=True, on_epoch=True)
      writer.add_scalar('val/loss_mean', loss_mean, self.trainer.global_step)
      writer.add_scalar('val/accuracy_mean', accuracy_mean, self.trainer.global_step)

  def test_step(self, train_batch, train_idx):
      preds = self.netD(train_batch[0])
      print("################")
      print(train_batch[1])
      print(preds.topk(k=1)[1])
