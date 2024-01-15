import random
from typing import Dict, Sequence, Tuple, Union, Any

import cv2
import numpy as np

from dicaugment.core.transforms_interface import DicomType

from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
    to_tuple,
    INTER_LINEAR,
)
from . import functional as F
from ..dicom import functional as Fdicom

__all__ = ["RandomScale", "LongestMaxSize", "SmallestMaxSize", "Resize"]


class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        scale_limit: float = 0.1,
        interpolation: int = INTER_LINEAR,
        always_apply=False,
        p=0.5,
    ):
        super(RandomScale, self).__init__(always_apply, p)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.interpolation = interpolation

    def get_params(self):
        """Returns parameters needed for the `apply` methods"""
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=INTER_LINEAR, **params):
        """Applies the transformation to the image"""
        return F.scale(img, scale, interpolation)

    def apply_to_bbox(self, bbox, **params):
        """Applies the transformation to a bbox. Bounding box coordinates are scale invariant"""
        return bbox

    def apply_to_keypoint(self, keypoint, scale=1, **params):
        """Applies the transformation to a keypoint"""
        return F.keypoint_scale(keypoint, scale, scale, scale)

    def apply_to_dicom(self, dicom: DicomType, scale=1, **params) -> DicomType:
        """Applies the augmentation to a dicom type"""
        return Fdicom.dicom_scale(dicom, scale, scale)

    def get_transform_init_args(self):
        """Returns initialization arguments (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1' : 1, 'arg2': 2))"""
        return {
            "interpolation": self.interpolation,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
        }


class LongestMaxSize(DualTransform):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of the image after the transformation. When using a list, max size
            will be randomly selected from the values in the list.
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(LongestMaxSize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int = 1024,
        interpolation: int = INTER_LINEAR,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.longest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox. Bounding box coordinates are scale invariant"""
        return bbox

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, max_size: int = 1024, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        height = params["rows"]
        width = params["cols"]
        depth = params["slices"]

        scale = max_size / max([height, width, depth])
        return F.keypoint_scale(keypoint, scale, scale, scale)

    def apply_to_dicom(
        self, dicom: DicomType, max_size: int = 1024, **params
    ) -> DicomType:
        """Applies the augmentation to a dicom type"""
        height = params["rows"]
        width = params["cols"]
        depth = params["slices"]
        scale = max_size / min([height, width, depth])
        return Fdicom.dicom_scale(dicom, scale, scale)

    def get_params(self) -> Dict[str, int]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "max_size": self.max_size
            if isinstance(self.max_size, int)
            else random.choice(self.max_size)
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("max_size", "interpolation")


class SmallestMaxSize(DualTransform):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of smallest side of the image after the transformation. When using a
            list, max size will be randomly selected from the values in the list.
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(SmallestMaxSize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.smallest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox. Bounding box coordinates are scale invariant"""
        return bbox

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, max_size: int = 1024, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        height = params["rows"]
        width = params["cols"]
        depth = params["slices"]

        scale = max_size / min([height, width, depth])
        return F.keypoint_scale(keypoint, scale, scale, scale)

    def apply_to_dicom(
        self, dicom: DicomType, max_size: int = 1024, **params
    ) -> DicomType:
        """Applies the augmentation to a dicom type"""
        height = params["rows"]
        width = params["cols"]
        depth = params["slices"]
        scale = max_size / min([height, width, depth])
        return Fdicom.dicom_scale(dicom, scale, scale)

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "max_size": self.max_size
            if isinstance(self.max_size, int)
            else random.choice(self.max_size)
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("max_size", "interpolation")


class Resize(DualTransform):
    """Resize the input to the given height, width, depth.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        depth (int): desired depth of the output.
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        height: int,
        width: int,
        depth: int,
        interpolation: int = INTER_LINEAR,
        always_apply=False,
        p=1,
    ):
        super(Resize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth
        self.interpolation = interpolation

    def apply(
        self, img: np.ndarray, interpolation: int = INTER_LINEAR, **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.resize(
            img,
            height=self.height,
            width=self.width,
            depth=self.depth,
            interpolation=interpolation,
        )

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox. Bounding box coordinates are scale invariant"""
        return bbox

    def apply_to_dicom(self, dicom: DicomType, **params) -> DicomType:
        """Applies the augmentation to a dicom type"""
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return Fdicom.dicom_scale(dicom, scale_x, scale_y)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        height = params["rows"]
        width = params["cols"]
        depth = params["slices"]
        scale_x = self.width / width
        scale_y = self.height / height
        scale_z = self.depth / depth
        return F.keypoint_scale(keypoint, scale_x, scale_y, scale_z)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("height", "width", "depth", "interpolation")
