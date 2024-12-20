from efficientnet_pytorch import EfficientNet

# from adamp import SGDP
import numpy as np
from statistics import mean
import pytorch_lightning as pl
import torch
from accuracy import calculate_accuracy
import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

from tensorboardX import SummaryWriter

writer = SummaryWriter(logdir=cfg["path"]["log_path"])


class CustomTrainClass(pl.LightningModule):
    def __init__(
        self,
        model_train="tf_efficientnetv2_b0",
        num_classes=3,
        diffaug_activate=False,
        policy="color,translation",
        aug=None,
    ):
        super().__init__()
        self.lr = cfg["lr"]
        #############################################
        if model_train == "efficientnet-b0":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b0", num_classes=num_classes
            )
        elif model_train == "efficientnet-b1":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b1", num_classes=num_classes
            )
        elif model_train == "efficientnet-b2":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b2", num_classes=num_classes
            )
        elif model_train == "efficientnet-b3":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b3", num_classes=num_classes
            )
        elif model_train == "efficientnet-b4":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b4", num_classes=num_classes
            )
        elif model_train == "efficientnet-b5":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b5", num_classes=num_classes
            )
        elif model_train == "efficientnet-b6":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b6", num_classes=num_classes
            )
        elif model_train == "efficientnet-b7":
            self.netD = EfficientNet.from_pretrained(
                "efficientnet-b7", num_classes=num_classes
            )

        elif model_train == "mobilenetv3_small":
            from arch.mobilenetv3_arch import MobileNetV3

            self.netD = MobileNetV3(n_class=num_classes, mode="small", input_size=256)
        elif model_train == "mobilenetv3_large":
            from arch.mobilenetv3_arch import MobileNetV3

            self.netD = MobileNetV3(n_class=num_classes, mode="large", input_size=256)

        elif model_train == "resnet50":
            from arch.resnet_arch import resnet50

            self.netD = resnet50(num_classes=num_classes, pretrain=cfg["pretrain"])
        elif model_train == "resnet101":
            from arch.resnet_arch import resnet101

            self.netD = resnet101(num_classes=num_classes, pretrain=cfg["pretrain"])
        elif model_train == "resnet152":
            from arch.resnet_arch import resnet152

            self.netD = resnet152(num_classes=num_classes, pretrain=cfg["pretrain"])

        #############################################
        elif model_train == "ViT":
            from vit_pytorch import ViT

            self.netD = ViT(
                image_size=256,
                patch_size=32,
                num_classes=num_classes,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
            )

        elif model_train == "DeepViT":
            from vit_pytorch.deepvit import DeepViT

            self.netD = DeepViT(
                image_size=256,
                patch_size=32,
                num_classes=num_classes,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
            )

        #############################################

        elif model_train == "RepVGG-A0":
            from arch.RepVGG_arch import create_RepVGG_A0

            self.netD = create_RepVGG_A0(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-A1":
            from arch.RepVGG_arch import create_RepVGG_A1

            self.netD = create_RepVGG_A1(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-A2":
            from arch.RepVGG_arch import create_RepVGG_A2

            self.netD = create_RepVGG_A2(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B0":
            from arch.RepVGG_arch import create_RepVGG_B0

            self.netD = create_RepVGG_B0(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B1":
            from arch.RepVGG_arch import create_RepVGG_B1

            self.netD = create_RepVGG_B1(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B1g2":
            from arch.RepVGG_arch import create_RepVGG_B1g2

            self.netD = create_RepVGG_B1g2(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B1g4":
            from arch.RepVGG_arch import create_RepVGG_B1g4

            self.netD = create_RepVGG_B1g4(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B2":
            from arch.RepVGG_arch import create_RepVGG_B2

            self.netD = create_RepVGG_B2(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B2g2":
            from arch.RepVGG_arch import create_RepVGG_B2g2

            self.netD = create_RepVGG_B2g2(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B2g4":
            from arch.RepVGG_arch import create_RepVGG_B2g4

            self.netD = create_RepVGG_B2g4(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B3":
            from arch.RepVGG_arch import create_RepVGG_B3

            self.netD = create_RepVGG_B3(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B3g2":
            from arch.RepVGG_arch import create_RepVGG_B3g2

            self.netD = create_RepVGG_B3g2(deploy=False, num_classes=num_classes)

        elif model_train == "RepVGG-B3g4":
            from arch.RepVGG_arch import create_RepVGG_B3g4

            self.netD = create_RepVGG_B3g4(deploy=False, num_classes=num_classes)

        #############################################

        elif model_train == "squeezenet_1_0":
            from arch.squeezenet_arch import SqueezeNet

            self.netD = SqueezeNet(num_classes=num_classes, version="1_0")

        elif model_train == "squeezenet_1_1":
            from arch.squeezenet_arch import SqueezeNet

            self.netD = SqueezeNet(num_classes=num_classes, version="1_1")
        #############################################
        elif model_train == "vgg11":
            from arch.vgg_arch import create_vgg11

            self.netD = create_vgg11(num_classes, pretrained=cfg["pretrain"])
        elif model_train == "vgg13":
            from arch.vgg_arch import create_vgg13

            self.netD = create_vgg13(num_classes, pretrained=cfg["pretrain"])
        elif model_train == "vgg16":
            from arch.vgg_arch import create_vgg16

            self.netD = create_vgg16(num_classes, pretrained=cfg["pretrain"])
        elif model_train == "vgg19":
            from arch.vgg_arch import create_vgg19

            self.netD = create_vgg19(num_classes, pretrained=cfg["pretrain"])

        #############################################
        elif model_train == "SwinTransformer":
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
                relative_pos_embedding=True,
            )

        elif model_train == "effV2":
            if cfg["model_size"] == "s":
                from arch.efficientnetV2_arch import effnetv2_s

                self.netD = effnetv2_s(num_classes=num_classes)
            elif cfg["model_size"] == "m":
                from arch.efficientnetV2_arch import effnetv2_m

                self.netD = effnetv2_m(num_classes=num_classes)
            elif cfg["model_size"] == "l":
                from arch.efficientnetV2_arch import effnetv2_l

                self.netD = effnetv2_l(num_classes=num_classes)
            elif cfg["model_size"] == "xl":
                from arch.efficientnetV2_arch import effnetv2_xl

                self.netD = effnetv2_xl(num_classes=num_classes)
            else:
                raise ValueError("model_size unknown")
        elif model_train == "x_transformers":
            from x_transformers import ViTransformerWrapper, Encoder

            self.netD = ViTransformerWrapper(
                image_size=cfg["image_size"],
                patch_size=cfg["patch_size"],
                num_classes=num_classes,
                attn_layers=Encoder(
                    dim=cfg["dim"],
                    depth=cfg["depth"],
                    heads=cfg["heads"],
                ),
            )

        elif model_train == "mobilevit":
            if cfg["model_size"] == "xxs":
                from arch.mobilevit_arch import mobilevit_xxs

                self.netD = mobilevit_xxs(num_classes=num_classes)
            elif cfg["model_size"] == "xs":
                from arch.mobilevit_arch import mobilevit_xs

                self.netD = mobilevit_xs(num_classes=num_classes)
            elif cfg["model_size"] == "x":
                from arch.mobilevit_arch import mobilevit_s

                self.netD = mobilevit_s(num_classes=num_classes)

        elif model_train == "hrt":
            from arch.hrt_arch import HighResolutionTransformer

            self.netD = HighResolutionTransformer(num_classes)

        elif model_train == "volo":
            if cfg["model_size"] == "volo_d1":
                from arch.volo_arch import volo_d1

                self.netD = volo_d1(pretrained=cfg["pretrain"], num_classes=num_classes)
            elif cfg["model_size"] == "volo_d2":
                from arch.volo_arch import volo_d2

                self.netD = volo_d2(pretrained=cfg["pretrain"], num_classes=num_classes)
            elif cfg["model_size"] == "volo_d3":
                from arch.volo_arch import volo_d3

                self.netD = volo_d3(pretrained=cfg["pretrain"], num_classes=num_classes)
            elif cfg["model_size"] == "volo_d4":
                from arch.volo_arch import volo_d4

                self.netD = volo_d4(pretrained=cfg["pretrain"], num_classes=num_classes)
            elif cfg["model_size"] == "volo_d5":
                from arch.volo_arch import volo_d5

                self.netD = volo_d5(pretrained=cfg["pretrain"], num_classes=num_classes)

        elif model_train == "pvt_v2":
            if cfg["model_size"] == "pvt_v2_b0":
                from arch.pvt_v2_arch import pvt_v2_b0

                self.netD = pvt_v2_b0(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "pvt_v2_b1":
                from arch.pvt_v2_arch import pvt_v2_b1

                self.netD = pvt_v2_b1(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "pvt_v2_b2":
                from arch.pvt_v2_arch import pvt_v2_b2

                self.netD = pvt_v2_b2(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "pvt_v2_b3":
                from arch.pvt_v2_arch import pvt_v2_b3

                self.netD = pvt_v2_b3(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "pvt_v2_b4":
                from arch.pvt_v2_arch import pvt_v2_b4

                self.netD = pvt_v2_b4(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "pvt_v2_b5":
                from arch.pvt_v2_arch import pvt_v2_b5

                self.netD = pvt_v2_b5(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "pvt_v2_b2_li":
                from arch.pvt_v2_arch import pvt_v2_b2_li

                self.netD = pvt_v2_b2_li(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )

        elif model_train == "ConvMLP":
            if cfg["model_size"] == "convmlp_s":
                from arch.ConvMLP_arch import convmlp_s

                self.netD = convmlp_s(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "convmlp_m":
                from arch.ConvMLP_arch import convmlp_m

                self.netD = convmlp_m(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )
            elif cfg["model_size"] == "convmlp_l":
                from arch.ConvMLP_arch import convmlp_l

                self.netD = convmlp_l(
                    pretrained=cfg["pretrain"], num_classes=num_classes
                )

        elif model_train == "FocalTransformer":
            from arch.focal_transformer_arch import FocalTransformer

            self.netD = FocalTransformer(num_classes=num_classes)

        elif model_train == "mobile_former":
            from arch.mobile_former_arch import (
                MobileFormer,
                config_52,
                config_294,
                config_508,
            )

            if cfg["model_size"] == "config_52":
                self.netD = MobileFormer(config_52)
            elif cfg["model_size"] == "config_294":
                self.netD = MobileFormer(config_294)
            elif cfg["model_size"] == "config_508":
                self.netD = MobileFormer(config_508)

        elif model_train == "poolformer":
            if cfg["model_size"] == "poolformer_s12":
                from arch.poolformer_arch import poolformer_s12

                self.netD = poolformer_s12(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "poolformer_s24":
                from arch.poolformer_arch import poolformer_s24

                self.netD = poolformer_s24(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "poolformer_s36":
                from arch.poolformer_arch import poolformer_s36

                self.netD = poolformer_s36(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "poolformer_m36":
                from arch.poolformer_arch import poolformer_m36

                self.netD = poolformer_m36(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "poolformer_m48":
                from arch.poolformer_arch import poolformer_m48

                self.netD = poolformer_m48(pretrained=True, num_classes=num_classes)

        elif model_train == "next_vit":
            if cfg["model_size"] == "small":
                from arch.next_vit_arch import nextvit_small

                self.netD = nextvit_small(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "base":
                from arch.next_vit_arch import nextvit_base

                self.netD = nextvit_base(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "large":
                from arch.next_vit_arch import nextvit_large

                self.netD = nextvit_large(pretrained=True, num_classes=num_classes)

        elif model_train == "hornet":
            if cfg["model_size"] == "hornet_tiny_7x7":
                from arch.hornet_arch import hornet_tiny_7x7

                self.netD = hornet_tiny_7x7(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "hornet_tiny_gf":
                from arch.hornet_arch import hornet_tiny_gf

                self.netD = hornet_tiny_gf(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "hornet_small_7x7":
                from arch.hornet_arch import hornet_small_7x7

                self.netD = hornet_small_7x7(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "hornet_small_gf":
                from arch.hornet_arch import hornet_small_gf

                self.netD = hornet_small_gf(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "hornet_base_7x7":
                from arch.hornet_arch import hornet_base_7x7

                self.netD = hornet_base_7x7(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "hornet_base_gf":
                from arch.hornet_arch import hornet_base_gf

                self.netD = hornet_base_gf(pretrained=True, num_classes=num_classes)

        elif model_train == "moganet":
            if cfg["model_size"] == "moganet_xtiny_1k":
                from arch.moganet_arch import moganet_xtiny_1k

                self.netD = moganet_xtiny_1k(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "moganet_tiny_1k":
                from arch.moganet_arch import moganet_tiny_1k

                self.netD = moganet_tiny_1k(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "moganet_tiny_1k_sz256":
                from arch.moganet_arch import moganet_tiny_1k_sz256

                self.netD = moganet_tiny_1k_sz256(
                    pretrained=True, num_classes=num_classes
                )
            if cfg["model_size"] == "moganet_small_1k":
                from arch.moganet_arch import moganet_small_1k

                self.netD = moganet_small_1k(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "moganet_base_1k":
                from arch.moganet_arch import moganet_base_1k

                self.netD = moganet_base_1k(pretrained=True, num_classes=num_classes)
            if cfg["model_size"] == "moganet_large_1k":
                from arch.moganet_arch import moganet_large_1k

                self.netD = moganet_large_1k(pretrained=True, num_classes=num_classes)

        elif model_train == "efficientvit":
            if cfg["model_size"] == "m0":
                from arch.efficientvit_arch import EfficientViT_M0

                self.netD = EfficientViT_M0(
                    pretrained="efficientvit_m0", num_classes=num_classes
                )
            if cfg["model_size"] == "m1":
                from arch.efficientvit_arch import EfficientViT_M1

                self.netD = EfficientViT_M1(
                    pretrained="efficientvit_m1", num_classes=num_classes
                )
            if cfg["model_size"] == "m2":
                from arch.efficientvit_arch import EfficientViT_M2

                self.netD = EfficientViT_M2(
                    pretrained="efficientvit_m2", num_classes=num_classes
                )
            if cfg["model_size"] == "m3":
                from arch.efficientvit_arch import EfficientViT_M3

                self.netD = EfficientViT_M3(
                    pretrained="efficientvit_m3", num_classes=num_classes
                )
            if cfg["model_size"] == "m4":
                from arch.efficientvit_arch import EfficientViT_M4

                self.netD = EfficientViT_M4(
                    pretrained="efficientvit_m4", num_classes=num_classes
                )
            if cfg["model_size"] == "m5":
                from arch.efficientvit_arch import EfficientViT_M5

                self.netD = EfficientViT_M5(
                    pretrained="efficientvit_m5", num_classes=num_classes
                )

        elif model_train == "timm":
            import timm

            self.netD = timm.create_model(
                cfg["model_choise"], num_classes=num_classes, pretrained=True
            )

        if cfg["compile"]:
            print("Using torch.compile")
            self.netD = torch.compile(self.netD)

        # weights_init(self.netD, 'kaiming') #only use this if there is no pretrain

        if aug == "gridmix":
            from GridMixupLoss import GridMixupLoss

            self.criterion = GridMixupLoss(
                alpha=(0.4, 0.7),
                hole_aspect_ratio=1.0,
                crop_area_ratio=(0.5, 1),
                crop_aspect_ratio=(0.5, 2),
                n_holes_x=(2, 6),
            )
        elif aug == "cutmix":
            from cutmix import cutmix

            self.criterion = cutmix(
                alpha=(0.4, 0.7),
                hole_aspect_ratio=1.0,
                crop_area_ratio=(0.5, 1),
                crop_aspect_ratio=(0.5, 2),
                n_holes_x=(2, 6),
            )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.accuracy = []
        self.losses = []
        self.diffaug_activate = diffaug_activate
        self.accuracy_val = []
        self.losses_val = []

        self.policy = policy

        if cfg["cutmix_or_mixup"]:
            from torchvision.transforms import v2

            cutmix = v2.CutMix(num_classes=2)
            mixup = v2.MixUp(num_classes=2)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def forward(self, x):
        return self.netD(x)

    def training_step(self, train_batch, batch_idx):
        input_data = train_batch[0]

        if cfg["ffcv"]:
            orig_labels = train_batch[1].view(-1)
        if not cfg["ffcv"]:
            orig_labels = train_batch[1]

        if cfg["cutmix_or_mixup"]:
            input_data, orig_labels = self.cutmix_or_mixup(input_data, orig_labels)

        preds = self.netD(input_data)

        loss = self.criterion(preds, orig_labels)
        writer.add_scalar("loss", loss, self.trainer.global_step)

        return loss

    def configure_optimizers(self):
        if cfg["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.netD.parameters(), lr=cfg["lr"])
        elif cfg["optimizer"] == "adamw_sf":
            from optimizer.adamw_sf import adamw_sf

            optimizer = adamw_sf(
                self.netD.parameters(),
                lr=self.lr,
            )
        elif cfg["optimizer"] == "adamw_win":
            from optimizer.adamw_win import adamw_win

            optimizer = adamw_win(
                self.netD.parameters(),
                lr=self.lr,
            )

        elif cfg["optimizer"] == "adan_sf":
            from optimizer.adan_sf import adan_sf

            optimizer = adan_sf(
                self.netD.parameters(),
                lr=self.lr,
            )
        elif cfg["optimizer"] == "adan":
            from optimizer.adan import adan

            optimizer = adan(
                self.netD.parameters(),
                lr=self.lr,
            )
        elif cfg["optimizer"] == "lamb":
            from optimizer.lamb import lamb

            optimizer = lamb(
                self.netD.parameters(),
                lr=self.lr,
            )
        return optimizer

    def on_training_epoch_end(self):
        if cfg["print_training_epoch_end_metrics"] is False:
            loss_mean = np.mean(self.losses)
            # accuracy_mean = torch.mean(self.accuracy)

        accuracy_mean = np.mean(self.accuracy)

        print(
            f"'Epoch': {self.current_epoch}, 'loss': {loss_mean:.2f}, 'accuracy': {accuracy_mean:.2f}"
        )

        # logging
        # self.log('train/loss_mean', loss_mean, prog_bar=True, logger=True, on_epoch=True)
        # self.log('train/accuracy_mean', accuracy_mean, prog_bar=True, logger=True, on_epoch=True)

        self.losses = []
        self.accuracy = []

    def validation_step(self, train_batch, train_idx):
        input_data = train_batch[0]

        if cfg["ffcv"]:
            orig_labels = train_batch[1].view(-1)
        if not cfg["ffcv"]:
            orig_labels = train_batch[1]

        preds = self.netD(input_data)

        loss = self.criterion(preds, orig_labels)
        self.losses_val.append(loss.item())

        self.accuracy_val.append(calculate_accuracy(preds, train_batch[1]).item())

    def on_validation_epoch_end(self):
        loss_mean = np.mean(self.losses_val)
        accuracy_mean = mean(self.accuracy_val)

        # print(f"'Epoch': {self.current_epoch}, 'loss_val': {loss_mean:.2f}, 'accuracy_val': {accuracy_mean:.2f}")
        # print("----------------------------------------------------------")

        # logging
        # self.log('val/loss_mean', loss_mean, prog_bar=True, logger=True, on_epoch=True)
        # self.log('val/accuracy_mean', accuracy_mean, prog_bar=True, logger=True, on_epoch=True)
        writer.add_scalar("val/loss_mean", loss_mean, self.trainer.global_step)
        writer.add_scalar("val/accuracy_mean", accuracy_mean, self.trainer.global_step)

        self.losses_val = []
        self.accuracy_val = []

        print(
            f"saving Checkpoint_{self.current_epoch}_{self.global_step}_loss_{loss_mean:3f}_acc_{accuracy_mean:3f}_D.pth"
        )
        torch.save(
            self.netD.state_dict(),
            f"Checkpoint_{self.current_epoch}_{self.global_step}_loss_{loss_mean:3f}_acc_{accuracy_mean:3f}_D.pth",
        )

    def test_step(self, train_batch, train_idx):
        preds = self.netD(train_batch[0])
        print(train_batch[1])
        print(preds.topk(k=1)[1])
