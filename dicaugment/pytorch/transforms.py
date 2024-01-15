from __future__ import absolute_import

import warnings

from typing import Sequence, Dict, List, Tuple, Any, Union, Callable
import numpy as np
import torch
from torchvision.transforms import functional as F
from . import functional as Ftorch

from ..core.transforms_interface import BasicTransform, BoxType, KeypointType

__all__ = ["ToPytorch"]

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

    def __init__(
        self,
        transpose_mask: bool = True,
        always_apply: bool = True,
        normalize: Union[None, Sequence[float]] = None,
        p=1.0,
    ):
        super(ToPytorch, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask
        self.normalize = normalize

    @property
    def targets(self) -> Dict[str, Callable]:
        """
        Returns the mapping of target to applicable `apply` method.
        (e.g. {'image': self.apply, 'bboxes', self.apply_to_bboxes})
        """
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img: np.ndarray, **params) -> torch.tensor:  # skipcq: PYL-W0613
        """Applies the transformation to the image"""
        if len(img.shape) not in [3, 4]:
            raise ValueError("DICaugment only supports images in HWD or HWDC format")

        if len(img.shape) == 3:
            return Ftorch.img_to_tensor(
                np.expand_dims(img, 3).transpose(3, 2, 0, 1), self.normalize
            )

        else:
            return Ftorch.img_to_tensor(img.transpose(3, 2, 0, 1), self.normalize)

    def apply_to_mask(
        self, mask: np.ndarray, **params
    ) -> torch.tensor:  # skipcq: PYL-W0613
        """Applies the augmentation to a mask"""
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        if self.transpose_mask and mask.ndim == 4:
            mask = mask.transpose(3, 2, 0, 1)
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks: List[np.ndarray], **params) -> List[torch.tensor]:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        return {}
