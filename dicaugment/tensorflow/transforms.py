from __future__ import absolute_import

import warnings

import numpy as np
import tensorflow as tf
from typing import Sequence, Dict, List, Tuple, Any, Union, Callable
from ..core.transforms_interface import BasicTransform

__all__ = ["ToTensorflow"]


class ToTensorflow(BasicTransform):
    """Convert image and mask to `Tensorflow.Tensor`. The numpy `HWDC` image is converted to Tensorflow `HWDC` tensor.
    If the image is in `HWD` format (grayscale image), it will be converted to Tensorflow `HWDC` tensor.

    Args:
        always_apply (bool): Indicates whether this transformation should be always applied. Default: True.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorflow, self).__init__(always_apply=always_apply, p=p)

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

    def apply(self, img: np.ndarray, **params) -> "tf.tensor":  # skipcq: PYL-W0613
        """Applies the transformation to the image"""
        if len(img.shape) not in [3, 4]:
            raise ValueError("DICaugment only supports images in HWD or HWDC format")

        if len(img.shape) == 3:
            return tf.convert_to_tensor(np.expand_dims(img, 3))

        else:
            return tf.convert_to_tensor(img)

    def apply_to_mask(
        self, mask: np.ndarray, **params
    ) -> "tf.tensor":  # skipcq: PYL-W0613
        """Applies the augmentation to a mask"""
        if mask.ndim == 3:
            mask = np.expand_dims(mask, 3)
        return tf.convert_to_tensor(mask)

    def apply_to_masks(self, masks: List[np.ndarray], **params) -> "tf.tensor":
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        return {}
