import argparse


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_train', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if args.model_train == 'efficientnet-b0':
      netD = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b1':
      netD = EfficientNet.from_pretrained('efficientnet-b1', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b2':
      netD = EfficientNet.from_pretrained('efficientnet-b2', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b3':
      netD = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b4':
      netD = EfficientNet.from_pretrained('efficientnet-b4', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b5':
      netD = EfficientNet.from_pretrained('efficientnet-b5', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b6':
      netD = EfficientNet.from_pretrained('efficientnet-b6', num_classes=args.num_classes)
    elif args.model_train == 'efficientnet-b7':
      netD = EfficientNet.from_pretrained('efficientnet-b7', num_classes=args.num_classes)



    elif args.model_train == 'mobilenetv3_small':
      from arch.mobilenetv3_arch import MobileNetV3
      netD = MobileNetV3(n_class=args.num_classes, mode='small', input_size=256)
    elif args.model_train == 'mobilenetv3_large':
      from arch.mobilenetv3_arch import MobileNetV3
      netD = MobileNetV3(n_class=args.num_classes, mode='large', input_size=256)



    elif args.model_train == 'resnet50':
      from arch.resnet_arch import resnet50
      netD = resnet50(num_classes=args.num_classes, pretrain=True)
    elif args.model_train == 'resnet101':
      from arch.resnet_arch import resnet101
      netD = resnet101(num_classes=args.num_classes, pretrain=True)
    elif args.model_train == 'resnet152':
      from arch.resnet_arch import resnet152
      netD = resnet152(num_classes=args.num_classes, pretrain=True)

    #############################################
    elif args.model_train == 'ViT':
      from vit_pytorch import ViT
      netD = ViT(
          image_size = 256,
          patch_size = 32,
          num_classes = args.num_classes,
          dim = 1024,
          depth = 6,
          heads = 16,
          mlp_dim = 2048,
          dropout = 0.1,
          emb_dropout = 0.1
      )

    elif args.model_train == 'DeepViT':
      from vit_pytorch.deepvit import DeepViT
      netD = DeepViT(
          image_size = 256,
          patch_size = 32,
          num_classes = args.num_classes,
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

    elif args.model_train == 'squeezenet_1_0':
      from arch.squeezenet_arch import SqueezeNet
      netD = SqueezeNet(num_classes=args.num_classes, version='1_0')

    elif args.model_train == 'squeezenet_1_1':
      from arch.squeezenet_arch import SqueezeNet
      netD = SqueezeNet(num_classes=args.num_classes, version='1_1')
    #############################################
    elif args.model_train == 'vgg11':
      from arch.vgg_arch import create_vgg11
      netD = create_vgg11(num_classes, pretrained=True)
    elif args.model_train == 'vgg13':
      from arch.vgg_arch import create_vgg13
      netD = create_vgg13(num_classes, pretrained=True)
    elif args.model_train == 'vgg16':
      from arch.vgg_arch import create_vgg16
      netD = create_vgg16(num_classes, pretrained=True)
    elif args.model_train == 'vgg19':
      from arch.vgg_arch import create_vgg19
      netD = create_vgg19(num_classes, pretrained=True)

    #############################################
    elif args.model_train == 'SwinTransformer':
      from swin_transformer_pytorch import SwinTransformer

      netD = SwinTransformer(
          hidden_dim=96,
          layers=(2, 2, 6, 2),
          heads=(3, 6, 12, 24),
          channels=3,
          num_classes=args.num_classes,
          head_dim=32,
          window_size=8,
          downscaling_factors=(4, 2, 2, 2),
          relative_pos_embedding=True
      )








    from torch.autograd import Variable

    import torch.onnx
    import torchvision
    import torch

    dummy_input = Variable(torch.randn(1, 3, 256, 256)) # don't set it too high, will run out of RAM
    state_dict = torch.load(args.model_path)
    print("Loaded model from model path into state_dict.")

    netD.load_state_dict(state_dict)
    torch.onnx.export(netD, dummy_input, args.output_path, opset_version=11)
    print("Done.")

if __name__ == "__main__":
    main()
