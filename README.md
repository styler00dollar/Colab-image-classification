# Colab-image-classification

Combines different architectures, losses, optimizers and augmentations for image classifications. Simply use `python train.py` after modifying the config file `config.yaml`. You can also do that inside Colab.

Dependencies:
```
pip install torch torchvision torchaudio --extra-index-url=https://download.pytorch.org/whl/cu124
pip install albumentations scikit-learn kornia efficientnet_pytorch x_transformers vit-pytorch swin-transformer-pytorch adamp tensorboardX torchvision timm madgrad pytorch_lightning git+https://github.com/styler00dollar/pytorch-randaugment adan-pytorch git+https://github.com/lilohuang/PyTurboJPEG.git ffcv ffcv_pl
MAX_JOBS=2 pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```
Build docker:
```
DOCKER_BUILDKIT=1 docker build -t styler00dollar/image_classification:latest .
```
Download docker:
```
docker pull styler00dollar/image_classification:latest
```
Run docker:
```
docker run --gpus all -it -v /path/:/workspace/ styler00dollar/image_classification:latest
```
