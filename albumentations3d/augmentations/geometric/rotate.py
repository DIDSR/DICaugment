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
    INTER_NEAREST
)
from ..crops import functional as FCrops
from . import functional as F

__all__ = [
    "Rotate",
    "RandomRotate90",
    #"SafeRotate"
    ]


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        axes (str, list of str): Defines the axis of rotation. Must be one of {'xy','yz','xz'} or a list of them.
            If a single str is passed, then all rotations will occur on that axis
            If a list is passed, then one axis of rotation will be chosen at random for each call of the transformation
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
                raise ValueError("Parameter axes must be one of {'xy','yz','xz'} or a list of these elements")
        elif isinstance(axes, Sequence) and len(set(axes).difference({"xy", "yz", "xz"})) != 0:
            raise ValueError("Parameter axes contains one or more elements that are not allowed. Got {}".format(set(axes).difference({"xy", "yz", "xz"})))
        
        self.axes = axes

    def apply(self, img: np.ndarray, factor: int = 0, axes: str = "xy", **params) -> np.ndarray:
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """

        return np.ascontiguousarray(np.rot90(img, factor, self.__str_axes_to_tuple[axes]))

    def get_params(self) -> Dict:
        # Random int in the range [0, 3]
        return {
            "factor": random.randint(0, 3),
            "axes" : self.axes if isinstance(self.axes, str) else random.choice(self.axes)
            }

    def apply_to_bbox(self, bbox: BoxInternalType, factor: int = 0, axes: str = "xy", **params) -> BoxInternalType:
        return F.bbox_rot90(bbox, factor, axes, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, factor: int = 0, axes: str = "xy", **params) -> KeypointInternalType:
        return F.keypoint_rot90(keypoint, factor, axes, **params)
    
    @property
    def __str_axes_to_tuple(self) -> Dict:
        return {
            "xy" : (0,1),
            "yz" : (0,2),
            "xz" : (1,2)
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("axes",)


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        axes (str, list of str): Defines the axis of rotation. Must be one of `{'xy','yz','xz'}` or a list of them.
            If a single str is passed, then all rotations will occur on that axis
            If a list is passed, then one axis of rotation will be chosen at random for each call of the transformation
        interpolation (int): scipy interpolation method (e.g. albumenations3d.INTER_NEAREST). Default: albumentations3d.INTER_LINEAR
        border_mode (str): scipy parameter to determine how the input image is extended during convolution or padding to maintain image shape
            Must be one of the following:
                `reflect` (d c b a | a b c d | d c b a)
                    The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
                `constant` (k k k k | a b c d | k k k k)
                    The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
                `nearest` (a a a a | a b c d | d d d d)
                    The input is extended by replicating the last pixel.
                `mirror` (d c b | a b c d | c b a)
                    The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
                `wrap` (a b c d | a b c d | a b c d)
                    The input is extended by wrapping around to the opposite edge.
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
        value (int or float): The fill value when border_mode = `constant`. Default: 0.
        mask_value (int, float,
                    list of ints,
                    list of float): The fill value when border_mode = `constant` applied for masks. Default: 0.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        crop_to_border (bool): If True, then the image is cropped to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Default: False
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
        value: Union[int,float] = 0,
        mask_value: Union[int,float] = 0,
        rotate_method: str = "largest_box",
        crop_to_border: bool = False,
        always_apply = False,
        p = 0.5,
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
                raise ValueError("Parameter axes must be one of {'xy','yz','xz'} or a list of these elements")
        elif isinstance(axes, Sequence) and len(set(axes).difference({"xy", "yz", "xz"})) != 0:
            raise ValueError("Parameter axes contains one or more elements that are not allowed. Got {}".format(set(axes).difference({"xy", "yz", "xz"})))
        if rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(self, img: np.ndarray, angle: float = 0, axes: str = "xy", **params) -> np.ndarray:
        return F.rotate(
            img,
            angle=angle,
            axes=axes,
            crop_to_border=self.crop_to_border,
            interpolation=self.interpolation,
            border_mode=self.border_mode,
            value=self.value
            )
    
    def apply_to_mask(self, img: np.ndarray, angle: float = 0, axes: str = "xy", **params) -> np.ndarray:
        return F.rotate(
            img,
            angle=angle,
            axes=axes,
            crop_to_border=self.crop_to_border,
            interpolation=INTER_NEAREST,
            border_mode=self.border_mode,
            value=self.mask_value
            )
        

    def apply_to_bbox(self, bbox: BoxInternalType, angle: float = 0, axes: str = "xy", cols: int = 0, rows: int = 0, slices: int = 0, **params) -> BoxInternalType:
        return F.bbox_rotate(
            bbox=bbox,
            angle=angle,
            method=self.rotate_method,
            axes=axes,
            crop_to_border=self.crop_to_border,
            rows=rows,
            cols=cols,
            slices=slices
        )

        # bbox_rotate(bbox: BoxInternalType, angle: float, method: str, axes: str, crop_to_border: bool, rows: int, cols: int, slices: int
        # bbox_out = F.bbox_rotate(bbox, angle, self.rotate_method, rows, cols)
        # if self.crop_border:
        #     bbox_out = FCrops.bbox_crop(bbox_out, x_min, y_min, x_max, y_max, rows, cols)
        # return bbox_out

    def apply_to_keypoint(self, keypoint: KeypointInternalType, angle: float = 0,  axes: str = "xy", cols: int = 0, rows: int = 0, slices: int = 0, **params) -> KeypointInternalType:
        
        return F.keypoint_rotate(
            keypoint = keypoint,
            angle=angle,
            axes=axes,
            crop_to_border=self.crop_to_border,
            rows=rows,
            cols=cols,
            slices=slices
        )
        # raise NotImplementedError()
        # keypoint_out = F.keypoint_rotate(keypoint, angle, rows, cols, **params)
        # if self.crop_border:
        #     keypoint_out = FCrops.crop_keypoint_by_coords(keypoint_out, (x_min, y_min, x_max, y_max))
        # return keypoint_out

    # @staticmethod
    # def _rotated_rect_with_max_area(h, w, angle):
    #     """
    #     Given a rectangle of size wxh that has been rotated by 'angle' (in
    #     degrees), computes the width and height of the largest possible
    #     axis-aligned rectangle (maximal area) within the rotated rectangle.

    #     Code from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    #     """

    #     angle = math.radians(angle)
    #     width_is_longer = w >= h
    #     side_long, side_short = (w, h) if width_is_longer else (h, w)

    #     # since the solutions for angle, -angle and 180-angle are all the same,
    #     # it is sufficient to look at the first quadrant and the absolute values of sin,cos:
    #     sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    #     if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
    #         # half constrained case: two crop corners touch the longer side,
    #         # the other two corners are on the mid-line parallel to the longer line
    #         x = 0.5 * side_short
    #         wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    #     else:
    #         # fully constrained case: crop touches all 4 sides
    #         cos_2a = cos_a * cos_a - sin_a * sin_a
    #         wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    #     return dict(
    #         x_min=max(0, int(w / 2 - wr / 2)),
    #         x_max=min(w, int(w / 2 + wr / 2)),
    #         y_min=max(0, int(h / 2 - hr / 2)),
    #         y_max=min(h, int(h / 2 + hr / 2)),
    #     )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "angle": random.uniform(self.limit[0], self.limit[1]),
            "axes" : self.axes if isinstance(self.axes, str) else random.choice(self.axes)
            }
        # if self.crop_border:
        #     h, w = params["image"].shape[:2]
        #     out_params.update(self._rotated_rect_with_max_area(h, w, out_params["angle"]))
        # return out_params

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("limit", "interpolation", "axes", "border_mode", "value", "mask_value", "rotate_method", "crop_to_border")


# class SafeRotate(DualTransform):
#     """Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

#     The resulting image may have artifacts in it. After rotation, the image may have a different aspect ratio, and
#     after resizing, it returns to its original shape with the original aspect ratio of the image. For these reason we
#     may see some artifacts.

#     Args:
#         limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
#             an angle is picked from (-limit, limit). Default: (-90, 90)
#         interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
#             cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
#             Default: cv2.INTER_LINEAR.
#         border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
#             cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
#             Default: cv2.BORDER_REFLECT_101
#         value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
#         mask_value (int, float,
#                     list of ints,
#                     list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
#         p (float): probability of applying the transform. Default: 0.5.

#     Targets:
#         image, mask, bboxes, keypoints

#     Image types:
#         uint8, float32
#     """

#     def __init__(
#         self,
#         limit: Union[float, Tuple[float, float]] = 90,
#         interpolation: int = cv2.INTER_LINEAR,
#         border_mode: int = cv2.BORDER_REFLECT_101,
#         value: FillValueType = None,
#         mask_value: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
#         always_apply: bool = False,
#         p: float = 0.5,
#     ):
#         super(SafeRotate, self).__init__(always_apply, p)
#         self.limit = to_tuple(limit)
#         self.interpolation = interpolation
#         self.border_mode = border_mode
#         self.value = value
#         self.mask_value = mask_value

#     def apply(self, img: np.ndarray, matrix: np.ndarray = np.array(None), **params) -> np.ndarray:
#         return F.safe_rotate(img, matrix, self.interpolation, self.value, self.border_mode)

#     def apply_to_mask(self, img: np.ndarray, matrix: np.ndarray = np.array(None), **params) -> np.ndarray:
#         return F.safe_rotate(img, matrix, cv2.INTER_NEAREST, self.mask_value, self.border_mode)

#     def apply_to_bbox(self, bbox: BoxInternalType, cols: int = 0, rows: int = 0, **params) -> BoxInternalType:
#         return F.bbox_safe_rotate(bbox, params["matrix"], cols, rows)

#     def apply_to_keypoint(
#         self,
#         keypoint: KeypointInternalType,
#         angle: float = 0,
#         scale_x: float = 0,
#         scale_y: float = 0,
#         cols: int = 0,
#         rows: int = 0,
#         **params
#     ) -> KeypointInternalType:
#         return F.keypoint_safe_rotate(keypoint, params["matrix"], angle, scale_x, scale_y, cols, rows)

#     @property
#     def targets_as_params(self) -> List[str]:
#         return ["image"]

#     def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
#         angle = random.uniform(self.limit[0], self.limit[1])

#         image = params["image"]
#         h, w = image.shape[:2]

#         # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
#         image_center = (w / 2, h / 2)

#         # Rotation Matrix
#         rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

#         # rotation calculates the cos and sin, taking absolutes of those.
#         abs_cos = abs(rotation_mat[0, 0])
#         abs_sin = abs(rotation_mat[0, 1])

#         # find the new width and height bounds
#         new_w = math.ceil(h * abs_sin + w * abs_cos)
#         new_h = math.ceil(h * abs_cos + w * abs_sin)

#         scale_x = w / new_w
#         scale_y = h / new_h

#         # Shift the image to create padding
#         rotation_mat[0, 2] += new_w / 2 - image_center[0]
#         rotation_mat[1, 2] += new_h / 2 - image_center[1]

#         # Rescale to original size
#         scale_mat = np.diag(np.ones(3))
#         scale_mat[0, 0] *= scale_x
#         scale_mat[1, 1] *= scale_y
#         _tmp = np.diag(np.ones(3))
#         _tmp[:2] = rotation_mat
#         _tmp = scale_mat @ _tmp
#         rotation_mat = _tmp[:2]

#         return {"matrix": rotation_mat, "angle": angle, "scale_x": scale_x, "scale_y": scale_y}

#     def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str]:
#         return ("limit", "interpolation", "border_mode", "value", "mask_value")
