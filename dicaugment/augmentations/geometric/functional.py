import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# import skimage.transform
import scipy.ndimage as ndimage
from functools import reduce

from dicaugment.augmentations.utils import (
    _maybe_process_in_chunks,
    _maybe_process_by_channel,
    angle_2pi_range,
    clipped,
    preserve_channel_dim,
    preserve_shape,
    SCIPY_MODE_TO_NUMPY_MODE,
)

from ... import random_utils
from ...core.bbox_utils import denormalize_bbox, normalize_bbox
from ...core.transforms_interface import (
    BoxInternalType,
    FillValueType,
    ImageColorType,
    KeypointInternalType,
    DicomType,
    INTER_NEAREST,
    INTER_LINEAR,
    INTER_QUADRATIC,
    INTER_CUBIC,
    INTER_QUARTIC,
    INTER_QUINTIC,
)

__all__ = [
    "pad",
    "pad_with_params",
    "bbox_rot90",
    "keypoint_rot90",
    "rotate",
    "bbox_rotate",
    "keypoint_rotate",
    "shift_scale_rotate",
    "keypoint_shift_scale_rotate",
    "bbox_shift_scale_rotate",
    "resize",
    "scale",
    "keypoint_scale",
    "py3round",
    "_func_max_size",
    "longest_max_size",
    "smallest_max_size",
    "bbox_flip",
    "bbox_hflip",
    "bbox_transpose",
    "bbox_vflip",
    "bbox_zflip",
    "hflip",
    "vflip",
    "zflip",
    "transpose",
    "keypoint_flip",
    "keypoint_hflip",
    "keypoint_transpose",
    "keypoint_vflip",
    "keypoint_zflip",
]


def bbox_rot90(
    bbox: BoxInternalType, factor: int, axes: str, rows: int, cols: int, slices: int
) -> BoxInternalType:  # skipcq: PYL-W0613
    """Rotates a bounding box by 90 degrees in the direction dicated by a right-handed coordinate system.
        i.e. from a top-level view of the xy plane:
            Rotation around the z-axis: counterclockwise rotation
            Rotation around the y-axis: left to right rotation
            Rotation around the x-axis: bottom to top rotation

    Args:
        bbox: A bounding box tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        axes: The axes that define the axis of rotation. Must be in {'xy','yz','xz'}
        rows: Image rows.
        cols: Image cols.
        slices: Image depth.

    Returns:
        tuple: A bounding box tuple (x_min, y_min, z_min, x_max, y_max, z_max).

    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    if axes not in {"xy", "yz", "xz"}:
        raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]

    if axes == "xy":
        if factor == 1:
            bbox = y_min, 1 - x_max, z_min, y_max, 1 - x_min, z_max
        elif factor == 2:
            bbox = 1 - x_max, 1 - y_max, z_min, 1 - x_min, 1 - y_min, z_max
        elif factor == 3:
            bbox = 1 - y_max, x_min, z_min, 1 - y_min, x_max, z_max
    elif axes == "xz":
        if factor == 1:
            bbox = 1 - z_max, y_min, x_min, 1 - z_min, y_max, z_min
        elif factor == 2:
            bbox = 1 - x_max, y_min, 1 - z_max, 1 - x_min, y_max, 1 - z_min
        elif factor == 3:
            bbox = z_min, y_min, 1 - x_min, z_max, y_max, 1 - x_max
    elif axes == "yz":
        if factor == 1:
            bbox = x_min, 1 - z_max, y_min, x_max, 1 - z_min, y_max
        elif factor == 2:
            bbox = x_min, 1 - y_max, 1 - z_max, x_max, 1 - y_min, 1 - z_min
        elif factor == 3:
            bbox = x_min, z_min, 1 - y_max, x_max, z_max, 1 - y_min

    return bbox


@angle_2pi_range
def keypoint_rot90(
    keypoint: KeypointInternalType,
    factor: int,
    axes: str,
    rows: int,
    cols: int,
    slices: int,
    **params,
) -> KeypointInternalType:
    """Rotates a bounding box by 90 degrees in the direction dicated by a right-handed coordinate system.
        i.e. from a top-level view of the xy plane;
            * Rotation around the z-axis; counterclockwise rotation
            * Rotation around the y-axis; left to right rotation
            * Rotation around the x-axis; bottom to top rotation

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.
        factor: Number of CCW rotations. Must be in range [0;3] See np.rot90.
        axes: The axes that define the axis of rotation. Must be in {'xy','yz','xz'}
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A keypoint `(x, y, z, angle, scale)`.

    Raises:
        ValueError: if factor not in set {0, 1, 2, 3}

    """
    x, y, z, angle, scale = keypoint[:5]

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    if axes not in {"xy", "yz", "xz"}:
        raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")

    if axes == "xy":
        if factor == 1:
            x, y, z, angle = y, (cols - 1) - x, z, angle - math.pi / 2
        elif factor == 2:
            x, y, z, angle = (cols - 1) - x, (rows - 1) - y, z, angle - math.pi
        elif factor == 3:
            x, y, z, angle = (rows - 1) - y, x, z, angle + math.pi / 2
    if axes == "xz":
        if factor == 1:
            x, y, z, angle = (slices - 1) - z, y, x, angle
        elif factor == 2:
            x, y, z, angle = (cols - 1) - x, y, (slices - 1) - z, angle
        elif factor == 3:
            x, y, z, angle = z, y, (cols - 1) - x, angle
    if axes == "yz":
        if factor == 1:
            x, y, z, angle = x, (slices - 1) - z, y, angle
        elif factor == 2:
            x, y, z, angle = x, (rows - 1) - y, (slices - 1) - z, angle
        elif factor == 3:
            x, y, z, angle = x, z, (rows - 1) - y, angle

    return x, y, z, angle, scale


def _get_new_image_shape(rows, cols, slices, rot_mat, scale_x=1, scale_y=1, scale_z=1):
    """
    Finds the bounding box of the image transformation and provides the new shape that encapsulates the entire image
    """
    rows /= 2
    cols /= 2
    slices /= 2
    arr = np.array(
        [
            [-rows, -cols, -slices],
            [-rows, -cols, slices],
            [-rows, cols, -slices],
            [-rows, cols, slices],
            [rows, -cols, -slices],
            [rows, -cols, slices],
            [rows, cols, -slices],
            [rows, cols, slices],
        ]
    ).T
    arr = np.round(np.matmul(rot_mat, arr))

    n_rows = int((np.max(arr[0]) - np.min(arr[0])) * scale_y)
    n_cols = int((np.max(arr[1]) - np.min(arr[1])) * scale_x)
    n_slices = int((np.max(arr[2]) - np.min(arr[2])) * scale_z)

    return n_rows, n_cols, n_slices


def _get_image_center(shape):
    return (np.array(shape) - 1) / 2


@preserve_channel_dim
def rotate(
    img: np.ndarray,
    angle: float,
    axes: str,
    crop_to_border: bool = False,
    interpolation: int = INTER_LINEAR,
    border_mode: int = "constant",
    value: Union[float, int] = 0,
):
    """
    Rotates an image by angle degrees.

    Args:
        img: Target image.
        angle: Angle of rotation in degrees.
        axes: The axis of rotation. Must be one of `{'xy', 'xz', 'yz'}`.
        crop_to_border: If True, then the image is cropped or padded to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Default: False
        interpolation: scipy interpolation method (e.g. dicaugment.INTER_NEAREST).
        border_mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        value: The fill value when border_mode = `constant`. Default: 0

    Returns:
        Image

    """

    out_shape = in_shape = img.shape[:3]
    height, width, depth = out_shape
    in_center = _get_image_center(in_shape)
    out_center = _get_image_center(out_shape)

    angle = np.deg2rad(angle)

    rotation_matrix = _get_rotation_matrix(angle, axes, dir=-1)

    if crop_to_border:
        out_shape = _get_new_image_shape(height, width, depth, rotation_matrix)
        out_center = _get_image_center(out_shape)

    matrix = np.linalg.inv(rotation_matrix)
    offset = in_center - np.dot(matrix, out_center)

    warp_affine_fn = _maybe_process_by_channel(
        ndimage.affine_transform,
        matrix=matrix,
        offset=offset,
        order=interpolation,
        output_shape=out_shape,
        mode=border_mode,
        cval=value,
    )

    return warp_affine_fn(img)


def bbox_rotate(
    bbox: BoxInternalType,
    angle: float,
    method: str,
    axes: str,
    crop_to_border: bool,
    rows: int,
    cols: int,
    slices: int,
) -> BoxInternalType:
    """Rotates a bounding box by angle degrees.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_min)`.
        angle: Angle of rotation in degrees.
        axes: The axis of rotation. Must be one of `{'xy', 'xz', 'yz'}`.
        crop_to_border: If True, bbox is normalized to fit new image shape. See `rotate(crop_to_border=True)`
        method: Rotation method used. Should be one of: "largest_box", "ellipse". Default: "largest_box".
        rows: Image rows.
        cols: Image cols.
        slices: Image slices

    Returns:
        A bounding box `(x_min, y_min, z_min, x_max, y_max, z_min)`.

    References:
        https://arxiv.org/abs/2109.13488

    """
    x_min, y_min, z_min, x_max, y_max, z_max = denormalize_bbox(
        list(map(lambda x: x - 0.5, bbox[:6])), rows, cols, slices
    )
    if method == "largest_box":
        bbox_points = np.array(
            [
                [y_min, x_min, z_min],
                [y_min, x_min, z_max],
                [y_max, x_min, z_min],
                [y_max, x_min, z_max],
                [y_min, x_max, z_min],
                [y_min, x_max, z_max],
                [y_max, x_max, z_min],
                [y_max, x_max, z_max],
            ]
        )

    elif method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        d = (z_max - z_min) / 2
        data = np.arange(0, 360, dtype=np.float32)

        if axes == "xy":
            x = np.tile(w * np.sin(np.radians(data)) + (w + x_min), 2)
            y = np.tile(h * np.cos(np.radians(data)) + (h + y_min), 2)
            z = np.concatenate((np.full((360,), z_min), np.full((360,), z_max)))
        elif axes == "xz":
            x = np.tile(w * np.sin(np.radians(data)) + (w + x_min), 2)
            y = np.concatenate((np.full((360,), y_min), np.full((360,), y_max)))
            z = np.tile(d * np.cos(np.radians(data)) + (d + z_min), 2)
        elif axes == "yz":
            x = np.concatenate((np.full((360,), x_min), np.full((360,), x_max)))
            y = np.tile(h * np.cos(np.radians(data)) + (h + y_min), 2)
            z = np.tile(d * np.sin(np.radians(data)) + (d + z_min), 2)
        else:
            raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")

        bbox_points = np.column_stack(
            [y, x, z],
        )
    else:
        raise ValueError(f"Method {method} is not a valid rotation method.")

    angle = np.deg2rad(angle)
    dir = 1

    if axes == "xy":
        dir = -1
    rotation_matrix = _get_rotation_matrix(angle, axes, dir=dir)

    bbox_points_t = np.matmul(rotation_matrix, bbox_points.T)

    x_min, x_max = np.min(bbox_points_t[1]), np.max(bbox_points_t[1])
    y_min, y_max = np.min(bbox_points_t[0]), np.max(bbox_points_t[0])
    z_min, z_max = np.min(bbox_points_t[2]), np.max(bbox_points_t[2])
    bbox = x_min, y_min, z_min, x_max, y_max, z_max

    if crop_to_border:
        rows, cols, slices = _get_new_image_shape(rows, cols, slices, rotation_matrix)

    return list(map(lambda x: x + 0.5, normalize_bbox(bbox, rows, cols, slices)))


@angle_2pi_range
def keypoint_rotate(
    keypoint,
    angle: float,
    axes: str,
    crop_to_border: bool,
    rows: int,
    cols: int,
    slices: int,
    **params,
):
    """Rotate a keypoint by angle.

    Args:
        keypoint (tuple): A keypoint `(x, y, z, angle, scale)`.
        angle (float): Rotation angle.
        axes: The axis of rotation. Must be one of `{'xy', 'xz', 'yz'}`.
        crop_to_border: If True, bbox is normalized to fit new image shape. See `rotate(crop_to_border=True)`
        rows (int): Image height.
        cols (int): Image width.
        slices: Image slices

    Returns:
        tuple: A keypoint `(x, y, z, angle, scale)`.

    """
    angle = np.deg2rad(angle)

    if axes == "xz":
        axes = "yz"

    elif axes == "yz":
        axes = "xz"

    in_center = _get_image_center((rows, cols, slices))
    out_center = in_center.copy()
    rotation_matrix = _get_rotation_matrix(angle, axes, dir=-1)

    if crop_to_border:
        out_shape = _get_new_image_shape(rows, cols, slices, rotation_matrix)
        out_center = _get_image_center(out_shape)

    x, y, z, a, s = keypoint[:5]

    p = np.array([[y, x, z]]) - in_center
    y, x, z = np.matmul(rotation_matrix, p.T).flatten() + out_center

    return x, y, z, a + (angle if axes == "xy" else 0), s


def _get_rotation_matrix(theta, axes, dir=1):
    if axes == "xy":
        arr = np.array(
            [
                [np.cos(theta), dir * np.sin(theta), 0],
                [dir * -np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    elif axes == "yz":
        arr = np.array(
            [
                [np.cos(theta), 0, dir * -np.sin(theta)],
                [0, 1, 0],
                [dir * np.sin(theta), 0, np.cos(theta)],
            ],
            dtype=np.float32,
        )

    elif axes == "xz":
        arr = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), dir * -np.sin(theta)],
                [0, dir * np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        )

    else:
        raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")

    return arr

def _get_scale_matrix(dx, dy, dz):
    return np.array(
        [
            [dy, 0, 0],
            [0, dx, 0],
            [0, 0, dz],
        ],
        dtype=np.float32,
    )


def _get_shear_matrix(mode, sx=None, sy=None, sz=None):
    if sum(list(map(lambda x: x is not None, [sx, sy, sz]))) != 2:
        raise ValueError(
            "Expected two of (sx,sy,sz) to be non-null arguments. Got sx={}, sy={}, sz={}".format(
                sx, sy, sz
            )
        )

    if mode == "xy":
        assert sx is not None and sy is not None

    elif mode == "xz":
        assert sx is not None and sz is not None

    elif mode == "yz":
        assert sy is not None and sz is not None

    else:
        raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")


@preserve_channel_dim
def shift_scale_rotate(
    img,
    angle,
    scale,
    dx,
    dy,
    dz,
    axes="xy",
    crop_to_border=False,
    interpolation=INTER_LINEAR,
    border_mode="constant",
    value=0,
):
    """
    Applies an affine transform to am image

    Args:
        img (np.ndarray): An image
        angle (float): an angle in degrees
        scale (float): the factpr to scale the image by
        dx (float): shift factor for width
        dy (float): shift factor for height
        dz (float): shift factor for depth
        axes (str): the axes to rotate along
        crop_to_border (bool): If True, then the image is padded or cropped to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Note that any translations are applied after the image is reshaped.
        interpolation: scipy interpolation method (e.g. dicaugment.INTER_NEAREST).
        border_mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        value: The fill value when border_mode = `constant`. Default: 0
    """
    out_shape = in_shape = img.shape[:3]
    height, width, depth = out_shape
    in_center = _get_image_center(in_shape)
    out_center = _get_image_center(out_shape)

    scale = (scale,) * 3
    angle = np.deg2rad(angle)

    rotation_matrix = _get_rotation_matrix(angle, axes, dir=-1)
    scale_matrix = _get_scale_matrix(*scale)

    if crop_to_border:
        out_shape = _get_new_image_shape(height, width, depth, rotation_matrix, *scale)
        out_center = _get_image_center(out_shape)

    matrix = np.linalg.inv(np.matmul(rotation_matrix, scale_matrix))
    offset = in_center - np.dot(matrix, out_center)
    shift = np.array([dy, dx, dz]) * np.array(out_shape)

    warp_affine_fn = _maybe_process_by_channel(
        ndimage.affine_transform,
        matrix=matrix,
        offset=offset,
        order=interpolation,
        output_shape=out_shape,
        mode=border_mode,
        cval=value,
    )
    translate_fn = _maybe_process_by_channel(
        ndimage.shift, shift=shift, order=interpolation, mode=border_mode, cval=value
    )
    return translate_fn(warp_affine_fn(img))


@angle_2pi_range
def keypoint_shift_scale_rotate(
    keypoint,
    angle,
    scale,
    dx,
    dy,
    dz,
    axes="xy",
    crop_to_border=False,
    rows=0,
    cols=0,
    slices=0,
    **params,
):
    """
    Applies an affine transform to a keypoint

    Args:
        keypoint (KeypointInternalType): A keypoint
        angle (float): an angle in degrees
        scale (float): the value to scale the keypoint's size by
        dx (float): shift factor for width
        dy (float): shift factor for height
        dz (float): shift factor for depth
        axes (str): the axes to rotate along
        crop_to_border (bool): If True, then the image is padded or cropped to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Note that any translations are applied after the image is reshaped.
        rows (int): Image width
        cols (int): Image height
        slices (int): Image depth
    """
    out_shape = height, width, depth = rows, cols, slices
    in_center = np.array(out_shape) / 2  # _get_image_center((rows, cols, slices))
    out_center = in_center.copy()
    scale = (scale,) * 3
    angle = np.deg2rad(angle)

    if axes == "xz":
        axes = "yz"
    elif axes == "yz":
        axes = "xz"

    rotation_matrix = _get_rotation_matrix(angle, axes, dir=-1)
    scale_matrix = _get_scale_matrix(*scale)

    if crop_to_border:
        out_shape = _get_new_image_shape(height, width, depth, rotation_matrix, *scale)
        out_center = _get_image_center(out_shape)

    matrix = np.matmul(rotation_matrix, scale_matrix)
    shift = np.array([dy, dx, dz]) * np.array(out_shape)

    x, y, z, a, s = keypoint[:5]

    p = np.array([[y, x, z]]) - in_center
    y, x, z = np.matmul(matrix, p.T).flatten() + out_center + shift

    return x, y, z, a + (angle if axes == "xy" else 0), s * scale[0]


def bbox_shift_scale_rotate(
    bbox,
    angle,
    scale,
    dx,
    dy,
    dz,
    axes="xy",
    crop_to_border=False,
    rotate_method="largest_box",
    rows=0,
    cols=0,
    slices=0,
    **kwargs,
):  # skipcq: PYL-W0613
    """Rotates, shifts and scales a bounding box. Rotation is made by angle degrees,
    scaling is made by scale factor and shifting is made by dx and dy.


    Args:
        bbox (tuple): A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        angle (int): Angle of rotation in degrees.
        scale (int): Scale factor.
        dx (int): Shift along x-axis.
        dy (int): Shift along y-axis.
        dz (int): Shift along z-axis.
        axes: The axis of rotation. Must be one of `{'xy', 'xz', 'yz'}`.
        crop_to_border: If True, bbox is normalized to fit new image shape. See `rotate(crop_to_border=True)`
        rotate_method(str): Rotation method used. Should be one of: "largest_box", "ellipse".
            Default: "largest_box".
        rows (int): Image rows.
        cols (int): Image cols.
        slices (int): Image slices

    Returns:
        A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """

    x_min, y_min, z_min, x_max, y_max, z_max = denormalize_bbox(
        list(map(lambda x: x - 0.5, bbox[:6])), rows, cols, slices
    )

    if rotate_method == "largest_box":
        bbox_points = np.array(
            [
                [y_min, x_min, z_min],
                [y_min, x_min, z_max],
                [y_max, x_min, z_min],
                [y_max, x_min, z_max],
                [y_min, x_max, z_min],
                [y_min, x_max, z_max],
                [y_max, x_max, z_min],
                [y_max, x_max, z_max],
            ]
        )

    elif rotate_method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        d = (z_max - z_min) / 2
        data = np.arange(0, 360, dtype=np.float32)

        if axes == "xy":
            x = np.tile(w * np.sin(np.radians(data)) + (w + x_min), 2)
            y = np.tile(h * np.cos(np.radians(data)) + (h + y_min), 2)
            z = np.concatenate((np.full((360,), z_min), np.full((360,), z_max)))
        elif axes == "yz":
            x = np.tile(w * np.sin(np.radians(data)) + (w + x_min), 2)
            y = np.concatenate((np.full((360,), y_min), np.full((360,), y_max)))
            z = np.tile(d * np.cos(np.radians(data)) + (d + z_min), 2)
        elif axes == "xz":
            x = np.concatenate((np.full((360,), x_min), np.full((360,), x_max)))
            y = np.tile(h * np.cos(np.radians(data)) + (h + y_min), 2)
            z = np.tile(d * np.sin(np.radians(data)) + (d + z_min), 2)
        else:
            raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")

        bbox_points = np.column_stack(
            [y, x, z],
        )
    else:
        raise ValueError(f"Method {rotate_method} is not a valid rotation method.")

    angle = np.deg2rad(angle)
    scale = (scale,) * 3
    dir = 1

    if axes == "xy":
        dir = -1
    rotation_matrix = _get_rotation_matrix(angle, axes, dir=dir)
    scale_matrix = _get_scale_matrix(*scale)
    shift = np.array([dx, dy, dz, dx, dy, dz])

    if crop_to_border:
        rows, cols, slices = _get_new_image_shape(
            rows, cols, slices, rotation_matrix, *scale
        )

    matrix = np.matmul(rotation_matrix, scale_matrix)

    bbox_points_t = np.matmul(matrix, bbox_points.T)

    x_min, x_max = np.min(bbox_points_t[1]), np.max(bbox_points_t[1])
    y_min, y_max = np.min(bbox_points_t[0]), np.max(bbox_points_t[0])
    z_min, z_max = np.min(bbox_points_t[2]), np.max(bbox_points_t[2])
    bbox = x_min, y_min, z_min, x_max, y_max, z_max

    # normalize points to assumed [0,1] range then apply translation
    return [
        p + s
        for s, p in zip(
            shift, map(lambda x: x + 0.5, normalize_bbox(bbox, rows, cols, slices))
        )
    ]


def _resize(img, dsize, interpolation):
    img_height, img_width, img_depth = img.shape[:3]
    dst_height, dst_width, dst_depth = dsize

    scale_y = dst_height / img_height
    scale_x = dst_width / img_width
    scale_z = dst_depth / img_depth

    return ndimage.zoom(img, zoom=(scale_y, scale_x, scale_z), order=interpolation)


@preserve_channel_dim
def resize(img: np.ndarray, height: int, width: int, depth: int , interpolation: int = INTER_LINEAR):
    """
    Resizes an image
    
    Args:
        img (np.ndarray): an image
        height (int): desired height of the output.
        width (int): desired width of the output.
        depth (int): desired depth of the output.
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
    """
    img_height, img_width, img_depth = img.shape[:3]
    if height == img_height and width == img_width and depth == img_depth:
        return img
    resize_fn = _maybe_process_by_channel(
        _resize, dsize=(height, width, depth), interpolation=interpolation
    )
    return resize_fn(img)


@preserve_channel_dim
def scale(
    img: np.ndarray,
    scale: Union[float, Tuple[float]],
    interpolation: int = INTER_LINEAR,
) -> np.ndarray:
    """
    Scales an image.

    Args:
        img (np.ndarray): an image
        scale (float, tuple): Scaling factor. If a tuple, then each dimension of image is scaled by respective element in tuple
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST). Default: dicaugment.INTER_LINEAR
    """
    scale_fn = _maybe_process_by_channel(ndimage.zoom, zoom=scale, order=interpolation)
    return scale_fn(img)


def keypoint_scale(
    keypoint: KeypointInternalType, scale_x: float, scale_y: float, scale_z: float
) -> KeypointInternalType:
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.
        scale_z: Scale coefficient y-axis.

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    """
    x, y, z, angle, scale = keypoint[:5]
    return (
        x * scale_x,
        y * scale_y,
        z * scale_z,
        angle,
        scale * max((scale_x, scale_y, scale_z)),
    )


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def _func_max_size(img, max_size, interpolation, func):
    height, width, depth = img.shape[:3]

    s = max_size / float(func((width, height, depth)))

    if s != 1.0:
        img = scale(img=img, scale=s, interpolation=interpolation)
    return img


@preserve_channel_dim
def longest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    """
    Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image
    
    Args:
        img (np.ndarray): an image
        max_size (int): the maximum side length of the image after resizing
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST)
    """
    return _func_max_size(img, max_size, interpolation, max)


@preserve_channel_dim
def smallest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    """
    Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
    
    Args:
        img (np.ndarray): an image
        max_size (int): the minimum side length of the image after resizing
        interpolation (int): scipy interpolation method (e.g. dicaugment.INTER_NEAREST)
    """
    return _func_max_size(img, max_size, interpolation, min)


def vflip(img: np.ndarray) -> np.ndarray:
    """Vertically flips an array"""
    return np.ascontiguousarray(img[::-1, ...])


def hflip(img: np.ndarray) -> np.ndarray:
    """Hortizontally flips an array"""
    return np.ascontiguousarray(img[:, ::-1, ...])


def zflip(img: np.ndarray) -> np.ndarray:
    """Flips an array along the slice dimension"""
    return np.ascontiguousarray(img[:, :, ::-1, ...])


def hflip_cv2(img: np.ndarray) -> np.ndarray:
    """Hortizontally flips an array using an OpenCV method"""
    return cv2.flip(img, 1)


@preserve_shape
def random_flip(img: np.ndarray, d: int) -> np.ndarray:
    if d == 0:
        img = vflip(img)
    elif d == 1:
        img = hflip(img)
    elif d == 2:
        img = zflip(img)
    elif d == -1:
        img = hflip(img)
        img = vflip(img)
        img = zflip(img)
    else:
        raise ValueError(
            "Invalid d value {}. Valid values are -1, 0, 1, and 2".format(d)
        )
    return img


def transpose(img: np.ndarray) -> np.ndarray:
    """Transposes an images dimensions"""
    return img.transpose(1, 0, 2, 3) if len(img.shape) > 3 else img.transpose(1, 0, 2)


def rot90(img: np.ndarray, factor: int, axes: Tuple[int,int] = (0, 1)) -> np.ndarray:
    """
    Rotates an image 90 degrees
    
    Args:
        img (np.ndarray): an image
        factor (int): the number of times to rotate the image
        axes (Tuple): the axes to rotate along

    """
    img = np.rot90(img, factor, axes)
    return np.ascontiguousarray(img)


def bbox_vflip(
    bbox: BoxInternalType, rows: int, cols: int, slices: int
) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box vertically around the x-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image rows.
        cols: Image cols.
        slices: Image slices

    Returns:
        tuple: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    return x_min, 1 - y_max, z_min, x_max, 1 - y_min, z_max


def bbox_hflip(
    bbox: BoxInternalType, rows: int, cols: int, slices: int
) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box horizontally around the y-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image rows.
        cols: Image cols.
        slices: Image slices

    Returns:
        A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    return 1 - x_max, y_min, z_min, 1 - x_min, y_max, z_max


def bbox_zflip(
    bbox: BoxInternalType, rows: int, cols: int, slices: int
) -> BoxInternalType:  # skipcq: PYL-W0613
    """Flip a bounding box on the z-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image rows.
        cols: Image cols.
        slices: Image slices

    Returns:
        A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    return x_min, y_min, 1 - z_max, x_max, y_max, 1 - z_min


def bbox_flip(
    bbox: BoxInternalType, d: int, rows: int, cols: int, slices: int
) -> BoxInternalType:
    """Flip a bounding box either vertically, horizontally, along the slice axis, or all depending on the value of `d`.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        d: dimension. 0 for vertical flip, 1 for horizontal, 2 for z-axis, -1 for transpose
        rows: Image rows.
        cols: Image cols.
        slices: Image slices

    Returns:
        A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    Raises:
        ValueError: if value of `d` is not -1, 0, 1, 2.

    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols, slices)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols, slices)
    elif d == 2:
        bbox = bbox_zflip(bbox, rows, cols, slices)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols, slices)
        bbox = bbox_vflip(bbox, rows, cols, slices)
        bbox = bbox_zflip(bbox, rows, cols, slices)
    else:
        raise ValueError(
            "Invalid d value {}. Valid values are -1, 0, 1, and 2".format(d)
        )
    return bbox


def bbox_transpose(
    bbox: KeypointInternalType, axis: int, rows: int, cols: int, slices: int
) -> KeypointInternalType:  # skipcq: PYL-W0613
    """Transposes a bounding box along given axis.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        axis: 0 - main axis, 1 - secondary axis.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box tuple `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    Raises:
        ValueError: If axis not equal to 0 or 1.

    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    if axis not in {0, 1}:
        raise ValueError("Parameter axes must be one of {0,1}")
    if axis == 0:
        bbox = (y_min, x_min, z_min, y_max, x_max, z_max)
    if axis == 1:
        bbox = (1 - y_max, 1 - x_max, z_min, 1 - y_min, 1 - x_min, z_max)
    return bbox


@angle_2pi_range
def keypoint_vflip(
    keypoint: KeypointInternalType, rows: int, cols: int, slices: int
) -> KeypointInternalType:
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.
        rows: Image height.
        cols: Image width.
        slices: Image depth

    Returns:
        tuple: A keypoint `(x, y, z, angle, scale)`.

    """
    x, y, z, angle, scale = keypoint[:5]
    angle = -angle
    return x, (rows - 1) - y, z, angle, scale


@angle_2pi_range
def keypoint_hflip(
    keypoint: KeypointInternalType, rows: int, cols: int, slices: int
) -> KeypointInternalType:
    """Flip a keypoint horizontally around the y-axis.

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.
        rows: Image height.
        cols: Image width.
        slices: Image depth

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    """
    x, y, z, angle, scale = keypoint[:5]
    angle = math.pi - angle
    return (cols - 1) - x, y, z, angle, scale


@angle_2pi_range
def keypoint_zflip(
    keypoint: KeypointInternalType, rows: int, cols: int, slices: int
) -> KeypointInternalType:
    """Flip a keypoint along the z-axis.

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.
        rows: Image height.
        cols: Image width.
        slices: Image depth

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    """
    x, y, z, angle, scale = keypoint[:5]
    return x, y, (slices - 1) - z, angle, scale


def keypoint_flip(
    keypoint: KeypointInternalType, d: int, rows: int, cols: int, slices: int
) -> KeypointInternalType:
    """Flip a keypoint either vertically, horizontally, along the slice axis, or all depending on the value of `d`.

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.
        d: Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * 2 - z-axis flip,
            * -1 - vertical, horizontal, and z-axis flip.
        rows: Image height.
        cols: Image width.
        slices: Image depth

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0, 1 or 2.

    """
    if d == 0:
        keypoint = keypoint_vflip(keypoint, rows, cols, slices)
    elif d == 1:
        keypoint = keypoint_hflip(keypoint, rows, cols, slices)
    elif d == 2:
        keypoint = keypoint_zflip(keypoint, rows, cols, slices)
    elif d == -1:
        keypoint = keypoint_hflip(keypoint, rows, cols, slices)
        keypoint = keypoint_vflip(keypoint, rows, cols, slices)
        keypoint = keypoint_zflip(keypoint, rows, cols, slices)
    else:
        raise ValueError(f"Invalid d value {d}. Valid values are -1, 0, 1, and 2")
    return keypoint


def keypoint_transpose(keypoint: KeypointInternalType) -> KeypointInternalType:
    """Rotate a keypoint by angle.

    Args:
        keypoint: A keypoint `(x, y, z, angle, scale)`.

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    """
    x, y, z, angle, scale = keypoint[:5]

    if angle <= np.pi:
        angle = np.pi - angle
    else:
        angle = 3 * np.pi - angle

    return y, x, z, angle, scale


@preserve_channel_dim
def pad(
    img: np.ndarray,
    min_height: int,
    min_width: int,
    min_depth: int,
    border_mode: int = "constant",
    value: Union[float, int] = 0,
) -> np.ndarray:
    """
    Pad an image.

    Args:
        img (np.ndarray): an image
        min_height (int): The minimum height to pad to, if applicable
        min_width (int): The minimum width to pad to, if applicable
        min_depth (int): The minimum depth to pad to, if applicable
        border_mode (str): Scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
            Default: `constant`.
        value (int or float): padding value if border_mode is "constant".
    """
    height, width, depth = img.shape[:3]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    if depth < min_depth:
        d_pad_close = int((min_depth - depth) / 2.0)
        d_pad_far = min_depth - depth - d_pad_close
    else:
        d_pad_close = 0
        d_pad_far = 0

    img = pad_with_params(
        img,
        h_pad_top,
        h_pad_bottom,
        w_pad_left,
        w_pad_right,
        d_pad_close,
        d_pad_far,
        border_mode,
        value,
    )

    if img.shape[:3] != (
        max(min_height, height),
        max(min_width, width),
        max(min_depth, depth),
    ):
        raise RuntimeError(
            "Invalid result shape. Got: {}. Expected: {}".format(
                img.shape[:3],
                (max(min_height, height), max(min_width, width), max(min_depth, depth)),
            )
        )

    return img


def _pad(
    img: np.ndarray,
    pad_width: Tuple[Tuple, Tuple, Tuple],
    border_mode: str = "constant",
    value: Union[float, int] = 0,
) -> np.ndarray:
    return np.pad(
        img,
        pad_width=pad_width,
        mode=SCIPY_MODE_TO_NUMPY_MODE[border_mode],
        constant_values=value,
    )


@preserve_channel_dim
def pad_with_params(
    img: np.ndarray,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    d_pad_front: int,
    d_pad_back: int,
    border_mode: str = "constant",
    value: Union[float, int] = None,
) -> np.ndarray:
    """
    Pad an image.

    Args:
        img (np.ndarray): an image
        h_pad_top (int): The number of pixels to pad on the top in the height dimension
        h_pad_bottom (int): The number of pixels to pad on the bottom in the height dimension
        w_pad_left (int): The number of pixels to pad on the left in the width dimension
        w_pad_right (int): The number of pixels to pad on the right in the width dimension
        d_pad_front (int): The number of pixels to pad on the front in the depth dimension
        d_pad_back (int): The number of pixels to pad on the back in the depth dimension
        border_mode (str): Scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
            Default: `constant`.
        value (int or float): padding value if border_mode is "constant".
    """
    pad_fn = _maybe_process_by_channel(
        _pad,
        pad_width=(
            (h_pad_top, h_pad_bottom),
            (w_pad_left, w_pad_right),
            (d_pad_front, d_pad_back),
        ),
        border_mode=border_mode,
        value=value,
    )
    return pad_fn(img)
