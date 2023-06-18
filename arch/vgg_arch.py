"""
4_vgg.ipynb (13-4-20)
https://github.com/styler00dollar/Colab-image-classification/blob/master/4_vgg.ipynb
"""
import torch
import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x  # , h


vgg11_config = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

vgg13_config = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

vgg16_config = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]

vgg19_config = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]


def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3

    for c in config:
        assert c == "M" or isinstance(c, int)
        if c == "M":
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)


"""
vgg11_layers = get_vgg_layers(vgg11_config, batch_norm = True)
OUTPUT_DIM = 10

model = VGG(vgg11_layers, OUTPUT_DIM)





pretrained_model = models.vgg11_bn(pretrained = True)

#print(pretrained_model)

pretrained_model.classifier[-1]

IN_FEATURES = pretrained_model.classifier[-1].in_features

final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

pretrained_model.classifier[-1] = final_fc

model.load_state_dict(pretrained_model.state_dict())
"""


def create_vgg11(num_classes, pretrained=True):
    vgg11_layers = get_vgg_layers(vgg16_config, batch_norm=True)
    model = VGG(vgg11_layers, num_classes)
    if pretrained == True:
        pretrained_model = models.vgg11_bn(pretrained=True)
        pretrained_model.classifier[-1]
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, num_classes)
        pretrained_model.classifier[-1] = final_fc
        model.load_state_dict(pretrained_model.state_dict())
        print("Loaded VGG11 pretrain.")
    return model


def create_vgg13(num_classes, pretrained=True):
    vgg16_layers = get_vgg_layers(vgg13_config, batch_norm=True)
    model = VGG(vgg13_layers, num_classes)
    if pretrained == True:
        pretrained_model = models.vgg13_bn(pretrained=True)
        pretrained_model.classifier[-1]
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, num_classes)
        pretrained_model.classifier[-1] = final_fc
        model.load_state_dict(pretrained_model.state_dict())
        print("Loaded VGG13 pretrain.")
    return model


def create_vgg16(num_classes, pretrained=True):
    vgg16_layers = get_vgg_layers(vgg16_config, batch_norm=True)
    model = VGG(vgg16_layers, num_classes)
    if pretrained == True:
        pretrained_model = models.vgg16_bn(pretrained=True)
        pretrained_model.classifier[-1]
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, num_classes)
        pretrained_model.classifier[-1] = final_fc
        model.load_state_dict(pretrained_model.state_dict())
        print("Loaded VGG16 pretrain.")
    return model


def create_vgg19(num_classes, pretrained=True):
    vgg19_layers = get_vgg_layers(vgg19_config, batch_norm=True)
    model = VGG(vgg19_layers, num_classes)
    if pretrained == True:
        pretrained_model = models.vgg19_bn(pretrained=True)
        pretrained_model.classifier[-1]
        IN_FEATURES = pretrained_model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, num_classes)
        pretrained_model.classifier[-1] = final_fc
        model.load_state_dict(pretrained_model.state_dict())
        print("Loaded VGG19 pretrain.")
    return model
