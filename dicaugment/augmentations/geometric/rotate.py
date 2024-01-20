import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    FillValueType,
    KeypointInternalType,
    to_tuple,
    INTER_LINEAR,
    INTER_NEAREST,
)
from ..crops import functional as FCrops
from . import functional as F

__all__ = [
    "Rotate",
    "RandomRotate90",
]


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        axes (str, list of str): Defines the axis of rotation. Must be one of {'xy','yz','xz'} or a list of them.
            If a single str is passed, then all rotations will occur on that axis
            If a list is passed, then one axis of rotation will be chosen at random for each call of the transformation
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        axes: str = "xy",
        always_apply=False,
        p=0.5,
    ):
        super(RandomRotate90, self).__init__(always_apply, p)
        if isinstance(axes, str):
            if axes not in {"xy", "yz", "xz"}:
                raise ValueError(
                    "Parameter axes must be one of {'xy','yz','xz'} or a list of these elements"
                )
        elif (
            isinstance(axes, Sequence)
            and len(set(axes).difference({"xy", "yz", "xz"})) != 0
        ):
            raise ValueError(
                "Parameter axes contains one or more elements that are not allowed. Got {}".format(
                    set(axes).difference({"xy", "yz", "xz"})
                )
            )

        self.axes = axes

    def apply(
        self, img: np.ndarray, factor: int = 0, axes: str = "xy", **params
    ) -> np.ndarray:
        """
        Applies the transformation to the image

        Args:
            img (np.ndarray): an image
            factor (int): number of times the input will be rotated by 90 degrees.
            axes (str): the axes to rotate along
        """

        return np.ascontiguousarray(
            np.rot90(img, factor, self.__str_axes_to_tuple[axes])
        )

    def get_params(self) -> Dict:
        """Returns parameters needed for the `apply` methods"""
        # Random int in the range [0, 3]
        return {
            "factor": random.randint(0, 3),
            "axes": self.axes
            if isinstance(self.axes, str)
            else random.choice(self.axes),
        }

    def apply_to_bbox(
        self, bbox: BoxInternalType, factor: int = 0, axes: str = "xy", **params
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_rot90(bbox, factor, axes, **params)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        factor: int = 0,
        axes: str = "xy",
        **params,
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_rot90(keypoint, factor, axes, **params)

    @property
    def __str_axes_to_tuple(self) -> Dict:
        return {"xy": (0, 1), "yz": (0, 2), "xz": (1, 2)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("axes",)


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        axes (str, list of str): Defines the axis of rotation. Must be one of `{'xy','yz','xz'}` or a list of them.
            If a single str is passed, then all rotations will occur on that axis
            If a list is passed, then one axis of rotation will be chosen at random for each call of the transformation
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        value (int or float): The fill value when border_mode = `constant`. Default: 0.
        mask_value (int, float, list of ints, list of float): The fill value when border_mode = `constant` applied for masks. Default: 0.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        crop_to_border (bool): If True, then the image is cropped to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Default: False
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        limit: float = 90,
        axes: str = "xy",
        interpolation: int = INTER_LINEAR,
        border_mode: str = "constant",
        value: Union[int, float] = 0,
        mask_value: Union[int, float] = 0,
        rotate_method: str = "largest_box",
        crop_to_border: bool = False,
        always_apply=False,
        p=0.5,
    ):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.axes = axes
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_to_border = crop_to_border

        if isinstance(axes, str):
            if axes not in {"xy", "yz", "xz"}:
                raise ValueError(
                    "Parameter axes must be one of {'xy','yz','xz'} or a list of these elements"
                )
        elif (
            isinstance(axes, Sequence)
            and len(set(axes).difference({"xy", "yz", "xz"})) != 0
        ):
            raise ValueError(
                "Parameter axes contains one or more elements that are not allowed. Got {}".format(
                    set(axes).difference({"xy", "yz", "xz"})
                )
            )
        if rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(
        self, img: np.ndarray, angle: float = 0, axes: str = "xy", **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.rotate(
            img,
            angle=angle,
            axes=axes,
            crop_to_border=self.crop_to_border,
            interpolation=self.interpolation,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self, img: np.ndarray, angle: float = 0, axes: str = "xy", **params
    ) -> np.ndarray:
        """Applies the transformation to a mask and forces INTER_NEAREST interpolation"""
        return F.rotate(
            img,
            angle=angle,
            axes=axes,
            crop_to_border=self.crop_to_border,
            interpolation=INTER_NEAREST,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        angle: float = 0,
        axes: str = "xy",
        cols: int = 0,
        rows: int = 0,
        slices: int = 0,
        **params,
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_rotate(
            bbox=bbox,
            angle=angle,
            method=self.rotate_method,
            axes=axes,
            crop_to_border=self.crop_to_border,
            rows=rows,
            cols=cols,
            slices=slices,
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        angle: float = 0,
        axes: str = "xy",
        cols: int = 0,
        rows: int = 0,
        slices: int = 0,
        **params,
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_rotate(
            keypoint=keypoint,
            angle=angle,
            axes=axes,
            crop_to_border=self.crop_to_border,
            rows=rows,
            cols=cols,
            slices=slices,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "angle": random.uniform(self.limit[0], self.limit[1]),
            "axes": self.axes
            if isinstance(self.axes, str)
            else random.choice(self.axes),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return (
            "limit",
            "interpolation",
            "axes",
            "border_mode",
            "value",
            "mask_value",
            "rotate_method",
            "crop_to_border",
        )
