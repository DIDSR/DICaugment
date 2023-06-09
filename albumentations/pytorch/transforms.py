from __future__ import absolute_import

import warnings

import numpy as np
import torch
from torchvision.transforms import functional as F
from . import functional as Ftorch

from ..core.transforms_interface import BasicTransform

__all__ = ["ToPytorch"]


# def img_to_tensor(im, normalize=None):
#     tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
#     if normalize is not None:
#         return F.normalize(tensor, **normalize)
#     return tensor


# def mask_to_tensor(mask, num_classes, sigmoid):
#     if num_classes > 1:
#         if not sigmoid:
#             # softmax
#             long_mask = np.zeros((mask.shape[:2]), dtype=np.int64)
#             if len(mask.shape) == 3:
#                 for c in range(mask.shape[2]):
#                     long_mask[mask[..., c] > 0] = c
#             else:
#                 long_mask[mask > 127] = 1
#                 long_mask[mask == 0] = 0
#             mask = long_mask
#         else:
#             mask = np.moveaxis(mask / (255.0 if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
#     else:
#         mask = np.expand_dims(mask / (255.0 if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
#     return torch.from_numpy(mask)


# class ToTensor(BasicTransform):
#     """Convert image and mask to `torch.Tensor` and divide by 255 if image or mask are `uint8` type.
#     This transform is now removed from Albumentations. If you need it downgrade the library to version 0.5.2.

#     Args:
#         num_classes (int): only for segmentation
#         sigmoid (bool, optional): only for segmentation, transform mask to LongTensor or not.
#         normalize (dict, optional): dict with keys [mean, std] to pass it into torchvision.normalize

#     """

#     def __init__(self, num_classes=1, sigmoid=True, normalize=None):
#         raise RuntimeError(
#             "`ToTensor` is obsolete and it was removed from Albumentations. Please use `ToTensorV2` instead - "
#             "https://albumentations.ai/docs/api_reference/pytorch/transforms/"
#             "#albumentations.pytorch.transforms.ToTensorV2. "
#             "\n\nIf you need `ToTensor` downgrade Albumentations to version 0.5.2."
#         )


class ToPytorch(BasicTransform):
    """Convert image and mask to `torch.Tensor`. The numpy `HWDC` image is converted to pytorch `CDHW` tensor.
    If the image is in `HWD` format (grayscale image), it will be converted to pytorch `DHW` tensor.

    Args:
        transpose_mask (bool): If True and an input mask has three spatial dimensions, this transform will transpose dimensions
            so the shape `[height, width, depth, channel]` becomes `[channel, depth, height, width]`. The latter format is a
            standard format for PyTorch Tensors. Default: True.
        always_apply (bool): Indicates whether this transformation should be always applied. Default: True.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, transpose_mask=True, always_apply=True, normalize = None, p=1.0):
        super(ToPytorch, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask
        self.normalize = normalize

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, "masks": self.apply_to_masks}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [3,4]:
            raise ValueError("Albumentations3D only supports images in HWD or HWDC format")

        if len(img.shape) == 3:
            return Ftorch.img_to_tensor(img.transpose(2, 0, 1), self.normalize)
        
        else:
            return Ftorch.img_to_tensor(img.transpose(3, 2, 0, 1), self.normalize)

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        if self.transpose_mask and mask.ndim == 4:
            mask = mask.transpose(3, 2, 0, 1)
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}
