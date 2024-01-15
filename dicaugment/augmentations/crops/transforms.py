import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from dicaugment.core.bbox_utils import union_of_bboxes

from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
    to_tuple,
    INTER_NEAREST,
    INTER_LINEAR,
    INTER_QUADRATIC,
    INTER_CUBIC,
    INTER_QUARTIC,
    INTER_QUINTIC,
)

from ..geometric import functional as FGeometric
from . import functional as F

__all__ = [
    "RandomCrop",
    "CenterCrop",
    "Crop",
    "RandomSizedCrop",
    "RandomCropNearBBox",
    "RandomSizedBBoxSafeCrop",
    "CropAndPad",
    "RandomCropFromBorders",
    "BBoxSafeRandomCrop",
]


class RandomCrop(DualTransform):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        depth (int): depth of the crop.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, int32, float32
    """

    def __init__(self, height: int, width: int, depth: int, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth

    def apply(
        self,
        img: np.ndarray,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.random_crop(
            img, self.height, self.width, self.depth, h_start, w_start, d_start
        )

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "d_start": random.random(),
        }

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_random_crop(bbox, self.height, self.width, self.depth, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_random_crop(
            keypoint, self.height, self.width, self.depth, **params
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("height", "width", "depth")


class CenterCrop(DualTransform):
    """Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        depth (int): depth of the crop.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, int32, float32
    """

    def __init__(self, height: int, width: int, depth: int, always_apply=False, p=1.0):
        super(CenterCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.center_crop(img, self.height, self.width, self.depth)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_center_crop(bbox, self.height, self.width, self.depth, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_center_crop(
            keypoint, self.height, self.width, self.depth, **params
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("height", "width", "depth")


class Crop(DualTransform):
    """Crop region from image.

    Args:
        x_min (int): Minimum closest upper left x coordinate.
        y_min (int): Minimum closest upper left y coordinate.
        z_min (int): Minimum closest upper left z coordinate.
        x_max (int): Maximum furthest lower right x coordinate.
        y_max (int): Maximum furthest lower right y coordinate.
        z_max (int): Maximum furthest lower right z coordinate.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, int32, float32
    """

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        z_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        z_max: int = 1024,
        always_apply=False,
        p=1.0,
    ):
        super(Crop, self).__init__(always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.crop(
            img,
            x_min=self.x_min,
            y_min=self.y_min,
            z_min=self.z_min,
            x_max=self.x_max,
            y_max=self.y_max,
            z_max=self.z_max,
        )

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_crop(
            bbox,
            x_min=self.x_min,
            y_min=self.y_min,
            z_min=self.z_min,
            x_max=self.x_max,
            y_max=self.y_max,
            z_max=self.z_max,
            **params
        )

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.crop_keypoint_by_coords(
            keypoint,
            crop_coords=(
                self.x_min,
                self.y_min,
                self.z_min,
                self.x_max,
                self.y_max,
                self.z_max,
            ),
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")


class _BaseRandomSizedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(
        self,
        height: int,
        width: int,
        depth: int,
        interpolation: int = INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):
        super(_BaseRandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_height: int = 0,
        crop_width: int = 0,
        crop_depth: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        interpolation: int = INTER_LINEAR,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        crop = F.random_crop(
            img, crop_height, crop_width, crop_depth, h_start, w_start, d_start
        )
        return FGeometric.resize(
            crop, self.height, self.width, self.depth, interpolation
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_height: int = 0,
        crop_width: int = 0,
        crop_depth: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        **params
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_random_crop(
            bbox,
            crop_height,
            crop_width,
            crop_depth,
            h_start,
            w_start,
            d_start,
            rows,
            cols,
            slices,
        )

    def apply_to_keypoint(
        self,
        keypoint,
        crop_height: int = 0,
        crop_width: int = 0,
        crop_depth: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        keypoint = F.keypoint_random_crop(
            keypoint,
            crop_height,
            crop_width,
            crop_depth,
            h_start,
            w_start,
            d_start,
            rows,
            cols,
            slices,
        )
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        scale_z = self.depth / crop_depth
        keypoint = FGeometric.keypoint_scale(keypoint, scale_x, scale_y, scale_z)
        return keypoint


class RandomSizedCrop(_BaseRandomSizedCrop):
    """Crop a random part of the input and rescale it to some size.

    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        depth (int): depth after crop and resize.
        w2h_ratio (float): width aspect ratio of crop.
        d2h_ratio (float): depth aspect ratio of crop.
        interpolation (int) : scipy interpolation method (e.g. dicaugment.INTER_NEAREST)
            Default: dicaugment.INTER_LINEAR.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        min_max_height: Tuple[int, int],
        height: int,
        width: int,
        depth: int,
        w2h_ratio: float = 1.0,
        d2h_ratio: float = 1.0,
        interpolation: int = INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):
        super(RandomSizedCrop, self).__init__(
            height=height,
            width=width,
            depth=depth,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio
        self.d2h_ratio = d2h_ratio

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        crop_height = random.randint(self.min_max_height[0], self.min_max_height[1])
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "crop_height": crop_height,
            "crop_width": int(crop_height * self.w2h_ratio),
            "crop_depth": int(crop_height * self.d2h_ratio),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return (
            "min_max_height",
            "height",
            "width",
            "depth",
            "w2h_ratio",
            "d2h_ratio",
            "interpolation",
        )

class RandomCropNearBBox(DualTransform):
    """Crop bbox from image with random shift by x,y,z coordinates

    Args:
        max_part_shift (float, (float, float, float)): Max shift in `height`, `width`, and `depth` dimensions relative
            to `cropping_bbox` dimension.
            If max_part_shift is a single float, the range will be (max_part_shift, max_part_shift, max_part_shift).
            Default (0.3, 0.3, 0.3).
        cropping_box_key (str): Additional target key for cropping box. Default `cropping_bbox`
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Examples:
        >>> aug = Compose([RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_box_key='test_box')],
        >>>              bbox_params=BboxParams("pascal_voc"))
        >>> result = aug(image=image, bboxes=bboxes, test_box=[0, 5, 10, 20])

    """

    def __init__(
        self,
        max_part_shift: Union[float, Tuple[float, float, float]] = (0.3, 0.3, 0.3),
        cropping_box_key: str = "cropping_bbox",
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(RandomCropNearBBox, self).__init__(always_apply, p)

        if isinstance(max_part_shift, float):
            self.max_part_shift = (max_part_shift,) * 3
        elif isinstance(max_part_shift, Sequence):
            if len(max_part_shift) != 3:
                raise ValueError(
                    "Expected max_part_shift to be a float or Tuple of length 3. Got {}".format(
                        max_part_shift
                    )
                )
            self.max_part_shift = max_part_shift
        else:
            raise ValueError(
                "Expected max_part_shift to be a float or Tuple. Got {}".format(
                    type(max_part_shift)
                )
            )

        self.cropping_bbox_key = cropping_box_key

        if min(self.max_part_shift) < 0 or max(self.max_part_shift) > 1:
            raise ValueError("Invalid max_part_shift. Got: {}".format(max_part_shift))

    def apply(
        self,
        img: np.ndarray,
        x_min: int = 0,
        y_min: int = 0,
        z_min: int = 0,
        x_max: int = 0,
        y_max: int = 0,
        z_max: int = 0,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.clamping_crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, int]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        bbox = params[self.cropping_bbox_key]
        h_max_shift = round((bbox[4] - bbox[1]) * self.max_part_shift[0])
        w_max_shift = round((bbox[3] - bbox[0]) * self.max_part_shift[1])
        d_max_shift = round((bbox[5] - bbox[2]) * self.max_part_shift[1])

        x_min = bbox[0] - random.randint(-w_max_shift, w_max_shift)
        x_max = bbox[3] + random.randint(-w_max_shift, w_max_shift)

        y_min = bbox[1] - random.randint(-h_max_shift, h_max_shift)
        y_max = bbox[4] + random.randint(-h_max_shift, h_max_shift)

        z_min = bbox[2] - random.randint(-d_max_shift, d_max_shift)
        z_max = bbox[5] + random.randint(-d_max_shift, d_max_shift)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)

        return {
            "x_min": x_min,
            "y_min": y_min,
            "z_min": z_min,
            "x_max": x_max,
            "y_max": y_max,
            "z_max": z_max,
        }

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_crop(bbox, **params)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        x_min: int = 0,
        y_min: int = 0,
        z_min: int = 0,
        x_max: int = 0,
        y_max: int = 0,
        z_max: int = 0,
        **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.crop_keypoint_by_coords(
            keypoint, crop_coords=(x_min, y_min, z_min, x_max, y_max, z_max)
        )

    @property
    def targets_as_params(self) -> List[str]:
        return [self.cropping_bbox_key]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("max_part_shift",)


class BBoxSafeRandomCrop(DualTransform):
    """
    Crop a random part of the input without loss of bboxes.

    Args:
        erosion_rate (float): erosion rate applied on input image height before crop.
        always_apply (bool): whether to always apply the transformation. Default: False.
        p (float): probability of applying the transform. Default: 1.
    
    Targets:
        image, mask, bboxes
    
    Image types:
        uint8, float32
    """

    def __init__(self, erosion_rate: float = 0.0, always_apply=False, p=1.0):
        super(BBoxSafeRandomCrop, self).__init__(always_apply, p)
        self.erosion_rate = erosion_rate

    def apply(
        self,
        img: np.ndarray,
        crop_height: int = 0,
        crop_width: int = 0,
        crop_depth: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.random_crop(
            img, crop_height, crop_width, crop_depth, h_start, w_start, d_start
        )

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        img_h, img_w, img_d = params["image"].shape[:3]
        if (
            len(params["bboxes"]) == 0
        ):  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = (
                img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            )
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "d_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
                "crop_depth": int(crop_height * img_d / img_h),
            }
        # get union of all bboxes
        x, y, z, x2, y2, z2 = union_of_bboxes(
            width=img_w,
            height=img_h,
            depth=img_d,
            bboxes=params["bboxes"],
            erosion_rate=self.erosion_rate,
        )
        # find bigger region
        bx, by, bz = x * random.random(), y * random.random(), z * random.random()
        bx2, by2, bz2 = (
            x2 + (1 - x2) * random.random(),
            y2 + (1 - y2) * random.random(),
            z2 + (1 - z2) * random.random(),
        )
        bw, bh, bd = bx2 - bx, by2 - by, bz2 - bz
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        crop_depth = img_d if bd >= 1.0 else int(img_d * bd)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        d_start = np.clip(0.0 if bd >= 1.0 else bz / (1.0 - bd), 0.0, 1.0)
        return {
            "h_start": h_start,
            "w_start": w_start,
            "d_start": d_start,
            "crop_height": crop_height,
            "crop_width": crop_width,
            "crop_depth": crop_depth,
        }

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_height: int = 0,
        crop_width: int = 0,
        crop_depth: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        **params
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_random_crop(
            bbox,
            crop_height,
            crop_width,
            crop_depth,
            h_start,
            w_start,
            d_start,
            rows,
            cols,
            slices,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", "bboxes"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("erosion_rate",)


class RandomSizedBBoxSafeCrop(BBoxSafeRandomCrop):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        depth (int): depth after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (int) : scipy interpolation method (e.g. dicaugment.INTER_NEAREST) Default: dicaugment.INTER_LINEAR.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height: int,
        width: int,
        depth: int,
        erosion_rate: float = 0.0,
        interpolation: int = INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):
        super(RandomSizedBBoxSafeCrop, self).__init__(erosion_rate, always_apply, p)
        self.height = height
        self.width = width
        self.depth = depth
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_height: int = 0,
        crop_width: int = 0,
        crop_depth: int = 0,
        h_start: int = 0,
        w_start: int = 0,
        d_start: int = 0,
        interpolation: int = INTER_LINEAR,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        crop = F.random_crop(
            img, crop_height, crop_width, crop_depth, h_start, w_start, d_start
        )
        return FGeometric.resize(
            crop, self.height, self.width, self.depth, interpolation
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return super().get_transform_init_args_names() + (
            "height",
            "width",
            "depth",
            "interpolation",
        )


class CropAndPad(DualTransform):
    """Crop and pad images by pixel amounts or fractions of image sizes.
    Cropping removes pixels at the sides (i.e. extracts a subimage from a given full image).
    Padding adds pixels to the sides (e.g. black pixels).
    This transformation will never crop images below a height or width of ``1``.

    Note:
        This transformation automatically resizes images back to their original size. To deactivate this, add the
        parameter ``keep_size=False``.

    Args:
        px (int or tuple):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image. Either this or the parameter `percent` may be set, not both at the same time.
                * If ``None``, then pixel-based cropping/padding will not be used.
                * If ``int``, then that exact number of pixels will always be cropped/padded.
                * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
                  then each side will be cropped/padded by a random amount sampled
                  uniformly per image and side from the interval ``[a, b]``. If
                  however `sample_independently` is set to ``False``, only one
                  value will be sampled per image and used for all sides.
                * If a ``tuple`` of six entries, then the entries represent top, bottom,
                  left, right, close, far. Each entry may be a single ``int`` (always
                  crop/pad by exactly that value), a ``tuple`` of two ``int`` s
                  ``a`` and ``b`` (crop/pad by an amount within ``[a, b]``), a
                  ``list`` of ``int`` s (crop/pad by a random value that is
                  contained in the ``list``).
        percent (float or tuple):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image given as a *fraction* of the image height/width. E.g. if this is set to ``-0.1``, the transformation will always crop away ``10%`` of the image's height at both the top and the bottom (both ``10%`` each), as well as ``10%`` of the width at the right and left. Expected value range is ``(-1.0, inf)``. Either this or the parameter `px` may be set, not both at the same time:
                * If ``None``, then fraction-based cropping/padding will not be used
                * If ``float``, then that fraction will always be cropped/padded
                * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``, then each side will be cropped/padded by a random fraction sampled uniformly per image and side from the interval ``[a, b]``. If however `sample_independently` is set to ``False``, only one value will be sampled per image and used for all sides.
                * If a ``tuple`` of six entries, then the entries represent top, bottom, left, right, close, far. Each entry may be a single ``float`` (always crop/pad by exactly that percent value), a ``tuple`` of two ``float`` s ``a`` and ``b`` (crop/pad by a fraction from ``[a, b]``), a ``list`` of ``float`` s (crop/pad by a random value that is contained in the list).
        pad_mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:
            - `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            - `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            - `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            - `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            - `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.
            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
            Default: `constant`
        pad_cval (number, Sequence[number]):
            The constant value to use if pad_mode is ``constant``.
                * If ``number``, then that value will be used.
                * If a ``tuple`` of two ``number`` s and at least one of them is
                  a ``float``, then a random number will be uniformly sampled per
                  image from the continuous interval ``[a, b]`` and used as the
                  value. If both ``number`` s are ``int`` s, the interval is
                  discrete.
                * If a ``list`` of ``number``, then a random value will be chosen
                  from the elements of the ``list`` and used as the value.
        pad_cval_mask (number, Sequence[number]): Same as pad_cval but only for masks.
        keep_size (bool):
            After cropping and padding, the result image will usually have a
            different height/width compared to the original input image. If this
            parameter is set to ``True``, then the cropped/padded image will be
            resized to the input image's size, i.e. the output shape is always identical to the input shape.
        sample_independently (bool):
            If ``False`` *and* the values for `px`/`percent` result in exactly
            *one* probability distribution for all image sides, only one single
            value will be sampled from that probability distribution and used for
            all sides. I.e. the crop/pad amount then is the same for all sides.
            If ``True``, four values will be sampled independently, one per side.
        interpolation (int) : scipy interpolation method (e.g. dicaugment.INTER_NEAREST)
            Default: dicaugment.INTER_LINEAR.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        any
    """

    def __init__(
        self,
        px: Optional[Union[int, Sequence[float], Sequence[Tuple]]] = None,
        percent: Optional[Union[float, Sequence[float], Sequence[Tuple]]] = None,
        pad_mode: str = "constant",
        pad_cval: Union[float, Sequence[float]] = 0,
        pad_cval_mask: Union[float, Sequence[float]] = 0,
        keep_size: bool = True,
        sample_independently: bool = True,
        interpolation: int = INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)

        if px is None and percent is None:
            raise ValueError("px and percent are empty!")
        if px is not None and percent is not None:
            raise ValueError("Only px or percent may be set!")

        self.px = px
        self.percent = percent

        self.pad_mode = pad_mode
        self.pad_cval = pad_cval
        self.pad_cval_mask = pad_cval_mask

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_params: Sequence[int] = (),
        pad_params: Sequence[int] = (),
        pad_value: Union[int, float] = 0,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        interpolation: int = INTER_LINEAR,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.crop_and_pad(
            img,
            crop_params,
            pad_params,
            pad_value,
            rows,
            cols,
            slices,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_mask(
        self,
        img: np.ndarray,
        crop_params: Optional[Sequence[int]] = None,
        pad_params: Optional[Sequence[int]] = None,
        pad_value_mask: Optional[float] = None,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        interpolation: int = INTER_NEAREST,
        **params
    ) -> np.ndarray:
        """Applies the transformation to a mask"""
        return F.crop_and_pad(
            img,
            crop_params,
            pad_params,
            pad_value_mask,
            rows,
            cols,
            slices,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_params: Optional[Sequence[int]] = None,
        pad_params: Optional[Sequence[int]] = None,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        result_rows: int = 0,
        result_cols: int = 0,
        result_slices: int = 0,
        **params
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.crop_and_pad_bbox(
            bbox,
            crop_params,
            pad_params,
            rows,
            cols,
            slices,
            result_rows,
            result_cols,
            result_slices,
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        crop_params: Optional[Sequence[int]] = None,
        pad_params: Optional[Sequence[int]] = None,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        result_rows: int = 0,
        result_cols: int = 0,
        result_slices: int = 0,
        **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.crop_and_pad_keypoint(
            keypoint,
            crop_params,
            pad_params,
            rows,
            cols,
            slices,
            result_rows,
            result_cols,
            result_slices,
            self.keep_size,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    @staticmethod
    def __prevent_zero(val1: int, val2: int, max_val: int) -> Tuple[int, int]:
        regain = abs(max_val) + 1
        regain1 = regain // 2
        regain2 = regain // 2
        if regain1 + regain2 < regain:
            regain1 += 1

        if regain1 > val1:
            diff = regain1 - val1
            regain1 = val1
            regain2 += diff
        elif regain2 > val2:
            diff = regain2 - val2
            regain2 = val2
            regain1 += diff

        val1 = val1 - regain1
        val2 = val2 - regain2

        return val1, val2

    @staticmethod
    def _prevent_zero(
        crop_params: List[int], height: int, width: int, depth: int
    ) -> Sequence[int]:
        top, bottom, left, right, close, far = crop_params

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)
        remaining_depth = depth - (close + far)

        if remaining_height < 1:
            top, bottom = CropAndPad.__prevent_zero(top, bottom, height)
        if remaining_width < 1:
            left, right = CropAndPad.__prevent_zero(left, right, width)
        if remaining_depth < 1:
            close, far = CropAndPad.__prevent_zero(close, far, depth)

        return [
            max(top, 0),
            max(bottom, 0),
            max(left, 0),
            max(right, 0),
            max(close, 0),
            max(far, 0),
        ]

    def get_params_dependent_on_targets(self, params) -> dict:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        height, width, depth = params["image"].shape[:3]

        if self.px is not None:
            params = self._get_px_params()
        else:
            params = self._get_percent_params()
            params[0] = int(params[0] * height)
            params[1] = int(params[1] * height)
            params[2] = int(params[2] * width)
            params[3] = int(params[3] * width)
            params[4] = int(params[4] * depth)
            params[5] = int(params[5] * depth)

        pad_params = [max(i, 0) for i in params]

        crop_params = self._prevent_zero(
            [-min(i, 0) for i in params], height, width, depth
        )

        top, bottom, left, right, close, far = crop_params
        crop_params = [top, height - bottom, left, width - right, close, depth - far]
        result_rows = crop_params[1] - crop_params[0]
        result_cols = crop_params[3] - crop_params[2]
        result_slices = crop_params[5] - crop_params[4]
        if result_cols == width and result_rows == height and result_slices == depth:
            crop_params = []

        top, bottom, left, right, close, far = pad_params
        if any(pad_params):
            result_rows += top + bottom
            result_cols += left + right
            result_slices += close + far
        else:
            pad_params = []

        return {
            "crop_params": crop_params or None,
            "pad_params": pad_params or None,
            "pad_value": None
            if pad_params is None
            else self._get_pad_value(self.pad_cval),
            "pad_value_mask": None
            if pad_params is None
            else self._get_pad_value(self.pad_cval_mask),
            "result_rows": result_rows,
            "result_cols": result_cols,
            "result_slices": result_slices,
        }

    def _get_px_params(self) -> List[int]:
        if self.px is None:
            raise ValueError("px is not set")

        if isinstance(self.px, int):
            params = [self.px] * 6
        elif len(self.px) == 2:
            if self.sample_independently:
                params = [random.randrange(*self.px) for _ in range(6)]
            else:
                px = random.randrange(*self.px)
                params = [px] * 6
        else:
            params = [i if isinstance(i, int) else random.randrange(*i) for i in self.px]  # type: ignore

        return params  # params = [left, right, top, bottom, close, far]

    def _get_percent_params(self) -> List[float]:
        if self.percent is None:
            raise ValueError("percent is not set")

        if isinstance(self.percent, float):
            params = [self.percent] * 6
        elif len(self.percent) == 2:
            if self.sample_independently:
                params = [random.uniform(*self.percent) for _ in range(6)]
            else:
                px = random.uniform(*self.percent)
                params = [px] * 6
        else:
            params = [
                i if isinstance(i, (int, float)) else random.uniform(*i)
                for i in self.percent
            ]

        return params  # params = [left, right, top, bottom, close, far]

    @staticmethod
    def _get_pad_value(pad_value: Union[float, Sequence[float]]) -> Union[int, float]:
        if isinstance(pad_value, (int, float)):
            return pad_value

        if len(pad_value) == 2:
            a, b = pad_value
            if isinstance(a, int) and isinstance(b, int):
                return random.randint(a, b)

            return random.uniform(a, b)

        return random.choice(pad_value)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return (
            "px",
            "percent",
            "pad_mode",
            "pad_cval",
            "pad_cval_mask",
            "keep_size",
            "sample_independently",
            "interpolation",
        )


class RandomCropFromBorders(DualTransform):
    """Crop bbox from image randomly cut parts from borders without resize at the end

    Args:
        crop_left (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
            from left side in range [0, crop_left * width)
        crop_right (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
            from right side in range [(1 - crop_right) * width, width)
        crop_top (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
            from top side in range [0, crop_top * height)
        crop_bottom (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
            from bottom side in range [(1 - crop_bottom) * height, height)
        crop_close (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
            from close side in range [0, crop_close * depth)
        crop_far (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
            from far side in range [(1 - crop_far) * depth, depth)
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        crop_close: float = 0.1,
        crop_far: float = 0.1,
        always_apply=False,
        p=1.0,
    ):
        super(RandomCropFromBorders, self).__init__(always_apply, p)
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_close = crop_close
        self.crop_far = crop_far

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        img = params["image"]
        x_min = random.randint(0, int(self.crop_left * img.shape[1]))
        x_max = random.randint(
            max(x_min + 1, int((1 - self.crop_right) * img.shape[1])), img.shape[1]
        )
        y_min = random.randint(0, int(self.crop_top * img.shape[0]))
        y_max = random.randint(
            max(y_min + 1, int((1 - self.crop_bottom) * img.shape[0])), img.shape[0]
        )
        z_min = random.randint(0, int(self.crop_close * img.shape[2]))
        z_max = random.randint(
            max(z_min + 1, int((1 - self.crop_far) * img.shape[2])), img.shape[2]
        )

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        }

    def apply(
        self,
        img: np.ndarray,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        z_min: int = 0,
        z_max: int = 0,
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.clamping_crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        z_min: int = 0,
        z_max: int = 0,
        **params
    ) -> np.ndarray:
        """Applies the transformation to a mask"""
        return F.clamping_crop(mask, x_min, y_min, z_min, x_max, y_max, z_max)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        z_min: int = 0,
        z_max: int = 0,
        **params
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        rows, cols, slices = params["rows"], params["cols"], params["slices"]
        return F.bbox_crop(
            bbox, x_min, y_min, z_min, x_max, y_max, z_max, rows, cols, slices
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        z_min: int = 0,
        z_max: int = 0,
        **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.crop_keypoint_by_coords(
            keypoint, crop_coords=(x_min, y_min, z_min, x_max, y_max, z_max)
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return (
            "crop_left",
            "crop_right",
            "crop_top",
            "crop_bottom",
            "crop_close",
            "crop_far",
        )
