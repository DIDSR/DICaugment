import random
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union, Any

import cv2
import numpy as np

from ...core.bbox_utils import denormalize_bbox, normalize_bbox
from ... import random_utils
from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    ImageColorType,
    KeypointInternalType,
    ScaleFloatType,
    DicomType,
    to_tuple,
    INTER_LINEAR,
    INTER_NEAREST,
)
from ..functional import bbox_from_mask
from . import functional as F
from ..dicom import functional as Fdicom

__all__ = [
    "ShiftScaleRotate",
    "VerticalFlip",
    "HorizontalFlip",
    "SliceFlip",
    "Flip",
    "Transpose",
    "PadIfNeeded",
]


class ShiftScaleRotate(DualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for height, width, and depth. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        axes (str, list of str): Defines the axis of rotation. Must be one of `{'xy','yz','xz'}` or a list of them.
            If a single str is passed, then all rotations will occur on that axis
            If a list is passed, then one axis of rotation will be chosen at random for each call of the transformation.
            Default: "xy"
        interpolation: scipy interpolation method (e.g. dicaugment.INTER_NEAREST). default: dicaugment.INTER_LINEAR
        border_mode (str): Scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
            Default: `constant`.
        value (int or float): padding value if border_mode is "constant".
        mask_value (int or float): padding value if border_mode is "constant" applied for masks.
        crop_to_border (bool): If True, then the image is padded or cropped to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Note that any translations are applied after the image is reshaped.
            Default: False
        shift_limit_x ((float, float) or float): shift factor range for width. If it is set then this value
                instead of shift_limit will be used for shifting width.  If shift_limit_x is a single float value,
                the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in
                the range [0, 1]. Default: None.
        shift_limit_y ((float, float) or float): shift factor range for height. If it is set then this value
            instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
            the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
            in the range [0, 1]. Default: None.
        shift_limit_z ((float, float) or float): shift factor range for depth. If it is set then this value
            instead of shift_limit will be used for shifting depth.  If shift_limit_z is a single float value,
            the range will be (-shift_limit_z, shift_limit_z). Absolute values for lower and upper bounds should lie
            in the range [0, 1]. Default: None.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        shift_limit: float = 0.0625,
        scale_limit: float = 0.1,
        rotate_limit: Union[int, float] = 45,
        axes: str = "xy",
        interpolation: int = INTER_LINEAR,
        border_mode: str = "constant",
        crop_to_border: bool = False,
        value: Union[int, float] = 0,
        mask_value: Union[int, float] = 0,
        shift_limit_x: Optional[Union[Tuple[float, float], float]] = None,
        shift_limit_y: Optional[Union[Tuple[float, float], float]] = None,
        shift_limit_z: Optional[Union[Tuple[float, float], float]] = None,
        rotate_method: str = "largest_box",
        always_apply=False,
        p=0.5,
    ):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit_x = to_tuple(
            shift_limit_x if shift_limit_x is not None else shift_limit
        )
        self.shift_limit_y = to_tuple(
            shift_limit_y if shift_limit_y is not None else shift_limit
        )
        self.shift_limit_z = to_tuple(
            shift_limit_z if shift_limit_z is not None else shift_limit
        )
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.crop_to_border = crop_to_border
        self.border_mode = border_mode
        self.value = value
        self.axes = (axes,) if isinstance(axes, str) else axes
        self.mask_value = mask_value
        self.rotate_method = rotate_method

        if self.rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")
        if len(set(self.axes).difference({"xy", "yz", "xz"})) != 0:
            raise ValueError(
                "Parameter axes contains one or more elements that are not allowed. Got {}".format(
                    set(self.axes).difference({"xy", "yz", "xz"})
                )
            )

    def apply(
        self,
        img: np.ndarray,
        angle: float = 0,
        axes: str = "xy",
        scale: float = 0,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        interpolation: int = INTER_LINEAR,
        **params,
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.shift_scale_rotate(
            img,
            angle,
            scale,
            dx,
            dy,
            dz,
            axes,
            self.crop_to_border,
            interpolation,
            self.border_mode,
            self.value,
        )

    def apply_to_mask(
        self,
        img: np.ndarray,
        angle: float = 0,
        axes: str = "xy",
        scale: float = 0,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        **params,
    ) -> np.ndarray:
        """Applies the transformation to a mask and forces INTER_NEAREST interpolation"""
        return F.shift_scale_rotate(
            img,
            angle,
            scale,
            dx,
            dy,
            dz,
            axes,
            self.crop_to_border,
            INTER_NEAREST,
            self.border_mode,
            self.mask_value,
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        angle: float = 0,
        axes: str = "xy",
        scale: float = 0,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        **params,
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_shift_scale_rotate(
            keypoint,
            angle,
            scale,
            dx,
            dy,
            dz,
            axes,
            self.crop_to_border,
            rows,
            cols,
            slices,
        )

    def apply_to_dicom(self, dicom: DicomType, scale: float = 1, **params) -> DicomType:
        """Applies the augmentation to a dicom type"""
        return Fdicom.dicom_scale(dicom, scale, scale)

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "dx": random.uniform(self.shift_limit_x[0], self.shift_limit_x[1]),
            "dy": random.uniform(self.shift_limit_y[0], self.shift_limit_y[1]),
            "dz": random.uniform(self.shift_limit_z[0], self.shift_limit_z[1]),
            "axes": random.choice(self.axes),
        }

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        angle: float = 0,
        axes: str = "xy",
        scale: float = 0,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        **params,
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_shift_scale_rotate(
            bbox,
            angle,
            scale,
            dx,
            dy,
            dz,
            axes,
            self.crop_to_border,
            self.rotate_method,
            **params,
        )

    def get_transform_init_args(self) -> Dict[str, Any]:
        """Returns initialization arguments (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1' : 1, 'arg2': 2))"""
        return {
            "shift_limit_x": self.shift_limit_x,
            "shift_limit_y": self.shift_limit_y,
            "shift_limit_z": self.shift_limit_z,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
            "rotate_limit": self.rotate_limit,
            "axes": self.axes,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
            "rotate_method": self.rotate_method,
        }

class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            or PositionType.RANDOM. Default: PositionType.CENTER.
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, list of float): padding value if border_mode is "constant".
        mask_value (int, float, list of int, list of float): padding value for mask if border_mode is "constant".
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    class PositionType(Enum):
        CENTER = "center"
        FRONT_TOP_LEFT = "front_top_left"
        FRONT_TOP_RIGHT = "front_top_right"
        FRONT_BOTTOM_LEFT = "front_bottom_left"
        FRONT_BOTTOM_RIGHT = "front_bottom_right"
        BACK_TOP_LEFT = "back_top_left"
        BACK_TOP_RIGHT = "back_top_right"
        BACK_BOTTOM_LEFT = "back_bottom_left"
        BACK_BOTTOM_RIGHT = "back_bottom_right"
        RANDOM = "random"

    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        min_depth: Optional[int] = 1024,
        pad_height_divisor: Optional[int] = None,
        pad_width_divisor: Optional[int] = None,
        pad_depth_divisor: Optional[int] = None,
        position: Union[PositionType, str] = PositionType.CENTER,
        border_mode: int = "constant",
        value: Union[float, int] = 0,
        mask_value: Union[float, int] = 0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError(
                "Only one of 'min_height' and 'pad_height_divisor' parameters must be set"
            )

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError(
                "Only one of 'min_width' and 'pad_width_divisor' parameters must be set"
            )

        if (min_depth is None) == (pad_depth_divisor is None):
            raise ValueError(
                "Only one of 'min_depth' and 'pad_depth_divisor' parameters must be set"
            )

        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.min_depth = min_depth
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor
        self.pad_depth_divisor = pad_depth_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]
        slices = params["slices"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = (
                self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0
            )

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        if self.min_depth is not None:
            if slices < self.min_depth:
                d_pad_front = int((self.min_depth - slices) / 2.0)
                d_pad_back = self.min_depth - slices - d_pad_front
            else:
                d_pad_front = 0
                d_pad_back = 0
        else:
            pad_remained = slices % self.pad_depth_divisor
            pad_slices = (
                self.pad_depth_divisor - pad_remained if pad_remained > 0 else 0
            )

            d_pad_front = pad_slices // 2
            d_pad_back = pad_slices - d_pad_front

        (
            h_pad_top,
            h_pad_bottom,
            w_pad_left,
            w_pad_right,
            d_pad_front,
            d_pad_back,
        ) = self.__update_position_params(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
            d_front=d_pad_front,
            d_back=d_pad_back,
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
                "pad_front": d_pad_front,
                "pad_back": d_pad_back,
            }
        )
        return params

    def apply(
        self,
        img: np.ndarray,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_front: int = 0,
        pad_back: int = 0,
        **params,
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            pad_front,
            pad_back,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self,
        img: np.ndarray,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_front: int = 0,
        pad_back: int = 0,
        **params,
    ) -> np.ndarray:
        """Applies the transformation to a mask"""
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            pad_front,
            pad_back,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_front: int = 0,
        pad_back: int = 0,
        rows: int = 0,
        cols: int = 0,
        slices: int = 0,
        **params,
    ) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        x_min, y_min, z_min, x_max, y_max, z_max = denormalize_bbox(
            bbox, rows, cols, slices
        )[:6]
        bbox = (
            x_min + pad_left,
            y_min + pad_top,
            z_min + pad_front,
            x_max + pad_left,
            y_max + pad_top,
            z_max + pad_front,
        )
        return normalize_bbox(
            bbox,
            rows + pad_top + pad_bottom,
            cols + pad_left + pad_right,
            slices + pad_front + pad_back,
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        pad_front: int = 0,
        pad_back: int = 0,
        **params,
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        x, y, z, angle, scale = keypoint[:5]
        return x + pad_left, y + pad_top, z + pad_front, angle, scale

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return (
            "min_height",
            "min_width",
            "min_depth",
            "pad_height_divisor",
            "pad_width_divisor",
            "pad_depth_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self,
        h_top: int,
        h_bottom: int,
        w_left: int,
        w_right: int,
        d_front: int,
        d_back: int,
    ) -> Tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.FRONT_TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            d_back += d_front

            h_top = 0
            w_left = 0
            d_front = 0

        elif self.position == PadIfNeeded.PositionType.BACK_TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            d_front += d_back

            h_top = 0
            w_left = 0
            d_back = 0

        elif self.position == PadIfNeeded.PositionType.FRONT_TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            d_back += d_front

            h_top = 0
            w_right = 0
            d_front = 0

        elif self.position == PadIfNeeded.PositionType.BACK_TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            d_front += d_back

            h_top = 0
            w_right = 0
            d_back = 0

        elif self.position == PadIfNeeded.PositionType.FRONT_BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            d_back += d_front

            h_bottom = 0
            w_left = 0
            d_front = 0

        elif self.position == PadIfNeeded.PositionType.BACK_BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            d_front += d_back

            h_bottom = 0
            w_left = 0
            d_back = 0

        elif self.position == PadIfNeeded.PositionType.FRONT_BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            d_back += d_front

            h_bottom = 0
            w_right = 0
            d_front = 0

        elif self.position == PadIfNeeded.PositionType.BACK_BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            d_front += d_back

            h_bottom = 0
            w_right = 0
            d_back = 0

        elif self.position == PadIfNeeded.PositionType.RANDOM:
            h_pad = h_top + h_bottom
            w_pad = w_left + w_right
            d_pad = d_front + d_back
            h_top = random.randint(0, h_pad)
            h_bottom = h_pad - h_top
            w_left = random.randint(0, w_pad)
            w_right = w_pad - w_left
            d_front = random.randint(0, d_pad)
            d_back = d_pad - d_front
        return h_top, h_bottom, w_left, w_right, d_front, d_back


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.vflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_vflip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_vflip(keypoint, **params)

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.hflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()


class SliceFlip(DualTransform):
    """Flip the input along the slice dimension.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.zflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_zflip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_zflip(keypoint, **params)

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()


class Flip(DualTransform):
    """Flip the input either horizontally, vertically, along the z-axis, or all.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, d: int = 0, **params) -> np.ndarray:
        """
        Applies the transformation to the image

        Args:
            img (np.ndarray): an image
            d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                    2 for z-axis flip, or -1 for vertical, horizontal, and z-axis flipping
        """
        return F.random_flip(img, d)

    def get_params(self):
        """Returns parameters needed for the `apply` methods"""
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 2)}

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns. Slice dimension remains unaffected

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.transpose(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the transformation to a bbox"""
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the transformation to a keypoint"""
        return F.keypoint_transpose(keypoint)

    def apply_to_dicom(self, dicom: DicomType, **params) -> DicomType:
        """Applies the augmentation to a dicom type"""
        return Fdicom.transpose_dicom(dicom)

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()
