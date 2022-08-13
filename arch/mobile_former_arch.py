"""
4-Dez-21
https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former/blob/main/model.py
https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former/blob/main/utils/mobile.py
https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former/blob/main/utils/former.py
https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former/blob/main/utils/bridge.py
https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former/blob/main/utils/config.py
https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former/blob/main/utils/utils.py
"""
import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


config_52 = {
    "name": "mf52",
    "token": 3,  # num tokens
    "embed": 128,  # embed dim
    "stem": 8,
    "bneck": {"e": 24, "o": 12, "s": 2},  # exp out stride
    "body": [
        # stage2
        {"inp": 12, "exp": 36, "out": 12, "se": None, "stride": 1, "heads": 2},
        # stage3
        {"inp": 12, "exp": 72, "out": 24, "se": None, "stride": 2, "heads": 2},
        {"inp": 24, "exp": 72, "out": 24, "se": None, "stride": 1, "heads": 2},
        # stage4
        {"inp": 24, "exp": 144, "out": 48, "se": None, "stride": 2, "heads": 2},
        {"inp": 48, "exp": 192, "out": 48, "se": None, "stride": 1, "heads": 2},
        {"inp": 48, "exp": 288, "out": 64, "se": None, "stride": 1, "heads": 2},
        # stage5
        {"inp": 64, "exp": 384, "out": 96, "se": None, "stride": 2, "heads": 2},
        {"inp": 96, "exp": 576, "out": 96, "se": None, "stride": 1, "heads": 2},
    ],
    "fc1": 1024,  # hid_layer
    "fc2": cfg["num_classes"],  # num_clasess
}

config_294 = {
    "name": "mf294",
    "token": 6,  # tokens
    "embed": 192,  # embed_dim
    "stem": 16,
    # stage1
    "bneck": {"e": 32, "o": 16, "s": 1},  # exp out stride
    "body": [
        # stage2
        {"inp": 16, "exp": 96, "out": 24, "se": None, "stride": 2, "heads": 2},
        {"inp": 24, "exp": 96, "out": 24, "se": None, "stride": 1, "heads": 2},
        # stage3
        {"inp": 24, "exp": 144, "out": 48, "se": None, "stride": 2, "heads": 2},
        {"inp": 48, "exp": 192, "out": 48, "se": None, "stride": 1, "heads": 2},
        # stage4
        {"inp": 48, "exp": 288, "out": 96, "se": None, "stride": 2, "heads": 2},
        {"inp": 96, "exp": 384, "out": 96, "se": None, "stride": 1, "heads": 2},
        {"inp": 96, "exp": 576, "out": 128, "se": None, "stride": 1, "heads": 2},
        {"inp": 128, "exp": 768, "out": 128, "se": None, "stride": 1, "heads": 2},
        # stage5
        {"inp": 128, "exp": 768, "out": 192, "se": None, "stride": 2, "heads": 2},
        {"inp": 192, "exp": 1152, "out": 192, "se": None, "stride": 1, "heads": 2},
        {"inp": 192, "exp": 1152, "out": 192, "se": None, "stride": 1, "heads": 2},
    ],
    "fc1": 1920,  # hid_layer
    "fc2": cfg["num_classes"],  # num_clasess
}

config_508 = {
    "name": "mf508",
    "token": 6,  # tokens and embed_dim
    "embed": 192,
    "stem": 24,
    "bneck": {"e": 48, "o": 24, "s": 1},
    "body": [
        {"inp": 24, "exp": 144, "out": 40, "se": None, "stride": 2, "heads": 2},
        {"inp": 40, "exp": 120, "out": 40, "se": None, "stride": 1, "heads": 2},
        {"inp": 40, "exp": 240, "out": 72, "se": None, "stride": 2, "heads": 2},
        {"inp": 72, "exp": 216, "out": 72, "se": None, "stride": 1, "heads": 2},
        {"inp": 72, "exp": 432, "out": 128, "se": None, "stride": 2, "heads": 2},
        {"inp": 128, "exp": 512, "out": 128, "se": None, "stride": 1, "heads": 2},
        {"inp": 128, "exp": 768, "out": 176, "se": None, "stride": 1, "heads": 2},
        {"inp": 176, "exp": 1056, "out": 176, "se": None, "stride": 1, "heads": 2},
        {"inp": 176, "exp": 1056, "out": 240, "se": None, "stride": 2, "heads": 2},
        {"inp": 240, "exp": 1440, "out": 240, "se": None, "stride": 1, "heads": 2},
        {"inp": 240, "exp": 1440, "out": 240, "se": None, "stride": 1, "heads": 2},
    ],
    "fc1": 1920,  # hid_layer
    "fc2": cfg["num_classes"],  # num_clasess
}

config = {"mf52": config_52, "mf294": config_294, "mf508": config_508}


import math
import torch
import random
import numpy as np
import torch.nn as nn


class MyDyRelu(nn.Module):
    def __init__(self, k):
        super(MyDyRelu, self).__init__()
        self.k = k

    def forward(self, inputs):
        x, relu_coefs = inputs
        # BxCxHxW -> HxWxBxCx1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # h w b c 1 -> h w b c k
        output = x_perm * relu_coefs[:, :, : self.k] + relu_coefs[:, :, self.k :]
        # HxWxBxCxk -> BxCxHxW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result


def mixup_data(x, y, alpha, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    b = x.size()[0]
    if use_cuda:
        index = torch.randperm(b).cuda()
    else:
        index = torch.randperm(b)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix(input, target, beta):
    lam = np.random.beta(beta, beta)
    b = input.size()[0]
    rand_index = torch.randperm(b).cuda()
    target_a = target
    target_b = target[rand_index]
    bx1, by1, bx2, by2 = rand_box(input.size(), lam)
    input[:, :, bx1:bx2, by1:by2] = input[rand_index, :, bx1:bx2, by1:by2]
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


def cutmix_criterion(criterion, output, target_a, target_b, lam):
    return lam * criterion(output, target_a) + (1.0 - lam) * criterion(output, target_b)


def rand_box(size, lam):
    _, _, h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    # 在图片上随机取一点作为cut的中心点
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bx1 = np.clip(cx - cut_w // 2, 0, w)
    by1 = np.clip(cy - cut_h // 2, 0, h)
    bx2 = np.clip(cx + cut_w // 2, 0, w)
    by2 = np.clip(cy + cut_h // 2, 0, h)
    return bx1, by1, bx2, by2


"""
for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
"""


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    """

    def __init__(
        self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            # 计算图片面积
            # c h w
            area = img.size()[1] * img.size()[2]
            # 比率范围
            target_area = random.uniform(self.sl, self.sh) * area
            # 宽高比
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                return img
        return img


import torch
from torch import nn, einsum
from einops import rearrange


# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.0):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel**-0.5
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        q = self.to_q(z).view(b, self.heads, m, c)
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ x
        out = rearrange(out, "b h m c -> b m (h c)")
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.0):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel**-0.5

        self.to_out = nn.Sequential(nn.Linear(inner_dim, channel), nn.Dropout(dropout))

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, "b h l c -> b l (h c)")
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out


import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):  # 2,65,1024 batch,patch+cls_token,dim (每个patch相当于一个token)
        b, n, _, h = *x.shape, self.heads
        # 输入x每个token的维度为1024，在注意力中token被映射16个64维的特征（head*dim_head），
        # 最后再把所有head的特征合并为一个（16*1024）的特征，作为每个token的输出
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv
        )  # 2,65,(16*64) -> 2,16,65,64 ,16个head，每个head维度64
        dots = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # b,16,65,64 @ b,16,64*65 -> b,16,65,65 : q@k.T
        attn = self.attend(
            dots
        )  # 注意力 2,16,65,65  16个head，注意力map尺寸65*65，对应token（patch）[i,j]之间的注意力
        # 每个token经过每个head的attention后的输出
        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # atten@v 2,16,65,65 @ 2,16,65,64 -> 2,16,65,64
        out = rearrange(
            out, "b h n d -> b n (h d)"
        )  # 合并所有head的输出(16*64) -> 1024 得到每个token当前的特征
        return self.to_out(out)


# inputs: n L C
# output: n L C
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        # dim_head = dim // heads
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.utils import MyDyRelu
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, inp, bias=False),
            hsigmoid(),
        )

    def forward(self, x):
        se = self.avg_pool(x)
        b, c, _, _ = se.size()
        se = se.view(b, c)
        se = self.se(se).view(b, c, 1, 1)
        return x * se.expand_as(x)


class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("lambdas", torch.Tensor([1.0] * k + [0.5] * k).float())
        self.register_buffer(
            "init_v", torch.Tensor([1.0] + [0.0] * (2 * k - 1)).float()
        )
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.conv1 = nn.Conv2d(inp, hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.act1 = MyDyRelu(2)

        self.conv2 = nn.Conv2d(
            hid,
            hid,
            kernel_size=ks,
            stride=stride,
            padding=ks // 2,
            groups=hid,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hid)
        self.act2 = MyDyRelu(2)

        self.conv3 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.bn1(self.conv1(x))
        out_ = [out, relu_coefs]
        out = self.act1(out_)

        out = self.bn2(self.conv2(out))
        out_ = [out, relu_coefs]
        out = self.act2(out_)

        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(MobileDown, self).__init__()
        self.dim = dim
        self.hid, self.out = hid, out
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("lambdas", torch.Tensor([1.0] * k + [0.5] * k).float())
        self.register_buffer(
            "init_v", torch.Tensor([1.0] + [0.0] * (2 * k - 1)).float()
        )
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.dw_conv1 = nn.Conv2d(
            inp,
            hid,
            kernel_size=ks,
            stride=stride,
            padding=ks // 2,
            groups=inp,
            bias=False,
        )
        self.dw_bn1 = nn.BatchNorm2d(hid)
        self.dw_act1 = MyDyRelu(2)

        self.pw_conv1 = nn.Conv2d(
            hid, inp, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.pw_bn1 = nn.BatchNorm2d(inp)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(
            inp, hid, kernel_size=ks, stride=1, padding=ks // 2, groups=inp, bias=False
        )
        self.dw_bn2 = nn.BatchNorm2d(hid)
        self.dw_act2 = MyDyRelu(2)

        self.pw_conv2 = nn.Conv2d(
            hid, out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.pw_bn2 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.dw_bn1(self.dw_conv1(x))
        out_ = [out, relu_coefs]
        out = self.dw_act1(out_)
        out = self.pw_act1(self.pw_bn1(self.pw_conv1(out)))

        out = self.dw_bn2(self.dw_conv2(out))
        out_ = [out, relu_coefs]
        out = self.dw_act2(out_)
        out = self.pw_bn2(self.pw_conv2(out))

        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


import time
import torch
import torch.nn as nn

from torch.nn import init

# from utils.mobile import Mobile, hswish, MobileDown
# from utils.former import Former
# from utils.bridge import Mobile2Former, Former2Mobile
# from utils.config import config_294, config_508, config_52


class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]


class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        self.token = nn.Parameter(
            nn.Parameter(torch.randn(1, cfg["token"], cfg["embed"]))
        )
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg["stem"], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg["stem"]),
            hswish(),
        )
        # bneck
        self.bneck = nn.Sequential(
            nn.Conv2d(
                cfg["stem"],
                cfg["bneck"]["e"],
                3,
                stride=cfg["bneck"]["s"],
                padding=1,
                groups=cfg["stem"],
            ),
            hswish(),
            nn.Conv2d(cfg["bneck"]["e"], cfg["bneck"]["o"], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg["bneck"]["o"]),
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg["body"]:
            self.block.append(BaseBlock(**kwargs, dim=cfg["embed"]))
        inp = cfg["body"][-1]["out"]
        exp = cfg["body"][-1]["exp"]
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(exp + cfg["embed"], cfg["fc1"]),
            hswish(),
            nn.Linear(cfg["fc1"], cfg["fc2"]),
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        for m in self.block:
            x, z = m([x, z])
        # x, z = self.block([x, z])
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        z = z[:, 0, :].view(b, -1)
        out = torch.cat((x, z), -1)
        return self.head(out)
        # return x, z


if __name__ == "__main__":
    model = MobileFormer(config_52)
    inputs = torch.randn((3, 3, 224, 224))
    print(inputs.shape)
    # for i in range(100):
    #     t = time.time()
    #     output = model(inputs)
    #     print(time.time() - t)
    print(
        "Total number of parameters in networks is {} M".format(
            sum(x.numel() for x in model.parameters()) / 1e6
        )
    )
    output = model(inputs)
    print(output.shape)
