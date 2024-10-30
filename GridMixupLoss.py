"""
gridmix_pytorch.py (27-3-20)
https://github.com/IlyaDobrynin/GridMixup/blob/main/gridmix/gridmix_pytorch.py
"""

import typing as t
import random
import numpy as np
import math
import torch
from torch import nn


class GridMixupLoss(nn.Module):
    """Implementation of GridMixup loss

    :param alpha: Percent of the first image on the crop. Can be float or Tuple[float, float]
                    - if float: lambda parameter gets from the beta-distribution np.random.beta(alpha, alpha)
                    - if Tuple[float, float]: lambda parameter gets from the uniform
                        distribution np.random.uniform(alpha[0], alpha[1])
    :param n_holes_x: Number of holes by OX
    :param hole_aspect_ratio: hole aspect ratio
    :param crop_area_ratio: Define percentage of the crop area
    :param crop_aspect_ratio: Define crop aspect ratio
    """

    def __init__(
        self,
        alpha: t.Union[float, t.Tuple[float, float]] = (0.1, 0.9),
        n_holes_x: t.Union[int, t.Tuple[int, int]] = 20,
        hole_aspect_ratio: t.Union[float, t.Tuple[float, float]] = 1.0,
        crop_area_ratio: t.Union[float, t.Tuple[float, float]] = 1.0,
        crop_aspect_ratio: t.Union[float, t.Tuple[float, float]] = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.n_holes_x = n_holes_x
        self.hole_aspect_ratio = hole_aspect_ratio
        self.crop_area_ratio = crop_area_ratio
        self.crop_aspect_ratio = crop_aspect_ratio
        if isinstance(self.n_holes_x, int):
            self.n_holes_x = (self.n_holes_x, self.n_holes_x)
        if isinstance(self.hole_aspect_ratio, float):
            self.hole_aspect_ratio = (self.hole_aspect_ratio, self.hole_aspect_ratio)
        if isinstance(self.crop_area_ratio, float):
            self.crop_area_ratio = (self.crop_area_ratio, self.crop_area_ratio)
        if isinstance(self.crop_aspect_ratio, float):
            self.crop_aspect_ratio = (self.crop_aspect_ratio, self.crop_aspect_ratio)

        self.loss = nn.CrossEntropyLoss()

    def __str__(self):
        return "gridmixup"

    @staticmethod
    def _get_random_crop(
        height: int, width: int, crop_area_ratio: float, crop_aspect_ratio: float
    ) -> t.Tuple:
        crop_area = int(height * width * crop_area_ratio)
        crop_width = int(np.sqrt(crop_area / crop_aspect_ratio))
        crop_height = int(crop_width * crop_aspect_ratio)

        cx = np.random.random()
        cy = np.random.random()

        y1 = int((height - crop_height) * cy)
        y2 = y1 + crop_height
        x1 = int((width - crop_width) * cx)
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    def _get_gridmask(
        self,
        image_shape: t.Tuple[int, int],
        crop_area_ratio: float,
        crop_aspect_ratio: float,
        lam: float,
        nx: int,
        ar: float,
    ) -> np.ndarray:
        """Method make grid mask

        :param image_shape: Shape of the images
        :param lam: Lambda parameter
        :param crop_area_ratio: Ratio of the crop area
        :param crop_aspect_ratio: Aspect ratio of the crop
        :param nx: Amount of holes by width
        :param ar: Aspect ratio of the hole
        :return: Binary mask, where holes == 1, background == 0
        """
        img_height, img_width = image_shape

        # Get coordinates of random box
        xc1, yc1, xc2, yc2 = self._get_random_crop(
            height=img_height,
            width=img_width,
            crop_area_ratio=crop_area_ratio,
            crop_aspect_ratio=crop_aspect_ratio,
        )
        height = yc2 - yc1
        width = xc2 - xc1

        if not 1 <= nx <= width // 2:
            raise ValueError(
                f"The nx must be between 1 and {width // 2}.\n" f"Give: {nx}"
            )

        # Get patch width, height and ny
        patch_width = math.ceil(width / nx)
        patch_height = int(patch_width * ar)
        ny = math.ceil(height / patch_height)

        # Calculate ratio of the hole - percent of hole pixels in the patch
        ratio = np.sqrt(1 - lam)

        # Get hole size
        hole_width = int(patch_width * ratio)
        hole_height = int(patch_height * ratio)

        # min 1 pixel and max patch length - 1
        hole_width = min(max(hole_width, 1), patch_width - 1)
        hole_height = min(max(hole_height, 1), patch_height - 1)

        # Make grid mask
        holes = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                x1 = min(patch_width * i, width)
                y1 = min(patch_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))

        mask = np.zeros(shape=image_shape, dtype=np.uint8)
        for x1, y1, x2, y2 in holes:
            mask[yc1 + y1 : yc1 + y2, xc1 + x1 : xc1 + x2] = 1
        return mask

    def get_sample(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Method returns augmented images and targets

        :param images: Batch of non-augmented images
        :param targets: Batch of non-augmented targets
        :return: Augmented images and targets
        """
        # Get new indices
        indices = torch.randperm(images.size(0)).to(images.device)

        # Shuffle labels
        shuffled_targets = targets[indices].to(targets.device)

        # Get image shape
        height, width = images.shape[2:]

        # Get lambda
        if isinstance(self.alpha, float):
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = np.random.uniform(self.alpha[0], self.alpha[1])

        nx = random.randint(self.n_holes_x[0], self.n_holes_x[1])
        ar = np.random.uniform(self.hole_aspect_ratio[0], self.hole_aspect_ratio[1])
        crop_area_ratio = np.random.uniform(
            self.crop_area_ratio[0], self.crop_area_ratio[1]
        )
        crop_aspect_ratio = np.random.uniform(
            self.crop_aspect_ratio[0], self.crop_aspect_ratio[1]
        )
        mask = self._get_gridmask(
            image_shape=(height, width),
            crop_area_ratio=crop_area_ratio,
            crop_aspect_ratio=crop_aspect_ratio,
            lam=lam,
            nx=nx,
            ar=ar,
        )
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - (mask.sum() / (images.size()[-1] * images.size()[-2]))

        # Make shuffled images
        mask = torch.from_numpy(mask).to(targets.device)
        images = images * (1 - mask) + images[indices, ...] * mask

        # Prepare out labels
        lam_list = torch.from_numpy(np.ones(shape=targets.shape) * lam).to(
            targets.device
        )
        out_targets = torch.cat([targets, shuffled_targets, lam_list], dim=1).transpose(
            0, 1
        )
        return images, out_targets

    def forward(self, preds: torch.Tensor, trues: torch.Tensor) -> torch.Tensor:
        lam = trues[-1][0].float()
        trues1, trues2 = trues[0].long(), trues[1].long()
        loss = self.loss(preds, trues1) * lam + self.loss(preds, trues2) * (1 - lam)
        return loss
