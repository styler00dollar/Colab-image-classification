# Colab-image-classification

Combines different architectures, losses, optimizers and augmentations for image classifications. Simply use `python train.py` after modifying the config file `config.yaml`. You can also do that inside Colab.

Dependencies:
```
pip install torch torchvision torchaudio --extra-index-url=https://download.pytorch.org/whl/cu124
pip install albumentations scikit-learn kornia efficientnet_pytorch x_transformers vit-pytorch swin-transformer-pytorch adamp tensorboardX torchvision timm madgrad pytorch_lightning git+https://github.com/styler00dollar/pytorch-randaugment adan-pytorch git+https://github.com/lilohuang/PyTurboJPEG.git ffcv ffcv_pl
```
