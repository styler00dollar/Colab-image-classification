# Colab-image-classification

Combines different architectures, losses, optimizers and augmentations for image classifications. Simply use `python train.py` after modifying the config file `config.yaml`. You can also do that inside Colab.

Dependencies:
```
!pip install kornia efficientnet_pytorch x_transformers vit-pytorch swin-transformer-pytorch adamp tensorboardX torchvision timm madgrad
pip install git+https://github.com/styler00dollar/pytorch-lightning.git@fc86f4ca817d5ba1702a210a898ac2729c870112
```
(You need to use this special pytorch lightning version, or training won't work.)

Old Colabs can be found in the `depricated` branch.
