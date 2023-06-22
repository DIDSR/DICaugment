import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
# import skimage.transform
import scipy.ndimage as ndimage
from functools import reduce

from albumentations3d.augmentations.utils import (
    _maybe_process_in_chunks,
    _maybe_process_by_channel,
    angle_2pi_range,
    clipped,
    preserve_channel_dim,
    preserve_shape,
    SCIPY_MODE_TO_NUMPY_MODE
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
    INTER_QUINTIC
)

__all__ = [
    #"optical_distortion",
    #"elastic_transform_approx",
    #"grid_distortion",
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
    #"elastic_transform",
    "resize",
    "scale",
    "keypoint_scale",
    "py3round",
    "_func_max_size",
    "longest_max_size",
    "smallest_max_size",
    #"perspective",
    #"perspective_bbox",
    #"rotation2DMatrixToEulerAngles",
    #"perspective_keypoint",
    #"_is_identity_matrix",
    #"warp_affine",
    #"keypoint_affine",
    #"bbox_affine",
    #"safe_rotate",
    #"bbox_safe_rotate",
    #"keypoint_safe_rotate",
    #"piecewise_affine",
    #"to_distance_maps",
    #"from_distance_maps",
    #"keypoint_piecewise_affine",
    #"bbox_piecewise_affine",
    "bbox_flip",
    "bbox_hflip",
    "bbox_transpose",
    "bbox_vflip",
    "bbox_zflip",
    "hflip",
    "vflip",
    "zflip",
    #"hflip_cv2",
    "transpose",
    "keypoint_flip",
    "keypoint_hflip",
    "keypoint_transpose",
    "keypoint_vflip",
    "keypoint_zflip",
]



def bbox_rot90(bbox: BoxInternalType, factor: int, axes: str, rows: int, cols: int, slices:int) -> BoxInternalType:  # skipcq: PYL-W0613
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

    if axes == 'xy':
        if factor == 1:
            bbox = y_min, 1 - x_max, z_min, y_max, 1 - x_min, z_max
        elif factor == 2:
            bbox = 1 - x_max, 1 - y_max, z_min, 1 - x_min, 1 - y_min, z_max
        elif factor == 3:
            bbox = 1 - y_max, x_min, z_min, 1 - y_min, x_max, z_max
    elif axes == 'xz':
        if factor == 1:
            bbox = 1 - z_max, y_min, x_min, 1 - z_min, y_max, z_min
        elif factor == 2:
            bbox = 1 - x_max, y_min, 1 - z_max, 1 - x_min, y_max, 1- z_min
        elif factor == 3:
            bbox = z_min, y_min, 1 - x_min, z_max, y_max, 1 - x_max
    elif axes == 'yz':
        if factor == 1:
            bbox = x_min, 1- z_max, y_min, x_max, 1 - z_min, y_max
        elif factor == 2:
            bbox = x_min, 1 - y_max, 1- z_max, x_max, 1 - y_min, 1 - z_min
        elif factor == 3:
            bbox = x_min, z_min, 1 - y_max, x_max, z_max, 1 - y_min
    
    return bbox


@angle_2pi_range
def keypoint_rot90(keypoint: KeypointInternalType, factor: int, axes: str, rows: int, cols: int, slices: int, **params) -> KeypointInternalType:
    """Rotates a bounding box by 90 degrees in the direction dicated by a right-handed coordinate system. 
        i.e. from a top-level view of the xy plane:
            Rotation around the z-axis: counterclockwise rotation
            Rotation around the y-axis: left to right rotation
            Rotation around the x-axis: bottom to top rotation
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

    if axes == 'xy':
        if factor == 1:
            x, y, z, angle = y, (cols - 1) - x, z, angle - math.pi / 2
        elif factor == 2:
            x, y, z, angle = (cols - 1) - x, (rows - 1) - y, z, angle - math.pi
        elif factor == 3:
            x, y, z, angle = (rows - 1) - y, x, z, angle + math.pi / 2
    if axes == 'xz':
        if factor == 1:
            x, y, z, angle = (slices - 1) - z, y, x, angle
        elif factor == 2:
            x, y, z, angle = (cols - 1) - x, y, (slices - 1) - z, angle
        elif factor == 3:
            x, y, z, angle = z, y, (cols - 1) - x, angle
    if axes == 'yz':
        if factor == 1:
            x, y, z, angle = x , (slices - 1) - z, y, angle
        elif factor == 2:
            x, y, z, angle = x, (rows - 1) - y, (slices - 1) - z, angle
        elif factor == 3:
            x, y, z, angle = x, z, (rows - 1) - y,  angle

    return x, y, z, angle, scale

def _get_new_image_shape(rows, cols, slices, rot_mat, scale_x = 1, scale_y = 1, scale_z = 1):
    """
    Finds the bounding box of the image transformation and provides the new shape that encapsulates the entire image
    """
    rows /= 2
    cols /= 2
    slices /= 2
    arr = np.array(
        [
            [-rows, -cols, -slices],
            [-rows, -cols,  slices],
            [-rows,  cols, -slices],
            [-rows,  cols,  slices],
            [ rows, -cols, -slices],
            [ rows, -cols,  slices],
            [ rows,  cols, -slices],
            [ rows,  cols,  slices],
        ]).T
    arr = np.round(np.matmul(rot_mat, arr))

    n_rows = int((np.max(arr[0]) - np.min(arr[0]))*scale_y)
    n_cols = int((np.max(arr[1]) - np.min(arr[1]))*scale_x)
    n_slices = int((np.max(arr[2]) - np.min(arr[2]))*scale_z)

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
    value: Union[float,int] = 0,
):
    """
    Rotates an image by angle degrees.
    
    Args:
        img: Target image.
        angle: Angle of rotation in degrees.
        axes: The axis of rotation. Must be one of `{'xy', 'xz', 'yz'}`.
        crop_to_border: If True, then the image is cropped or padded to fit the entire rotation. If False, then original image shape is
            maintained and some portions of the image may be cropped away. Default: False
        interpolation: scipy interpolation method (e.g. albumenations3d.INTER_NEAREST).
        border_mode: scipy parameter to determine how the input image is extended during convolution to maintain image shape
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
            Default: `constant`
        value: The fill value when border_mode = `constant`. Default: 0        

    Returns:
        Image

    """
    
    out_shape = in_shape =  img.shape[:3]
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
        ndimage.affine_transform, matrix= matrix, offset = offset, order=interpolation, output_shape=out_shape, mode=border_mode, cval=value
    )

    return warp_affine_fn(img)


def bbox_rotate(bbox: BoxInternalType, angle: float, method: str, axes: str, crop_to_border: bool, rows: int, cols: int, slices: int) -> BoxInternalType:
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
    x_min, y_min, z_min, x_max, y_max, z_max = denormalize_bbox(list(map(lambda x: x - 0.5, bbox[:6])), rows, cols, slices)
    if method == "largest_box":
        bbox_points = np.array(
            [
                [y_min, x_min,  z_min],
                [y_min, x_min,  z_max],
                [y_max, x_min,  z_min],
                [y_max, x_min,  z_max],
                [y_min, x_max,  z_min],
                [y_min, x_max,  z_max],
                [y_max, x_max,  z_min],
                [y_max, x_max,  z_max]
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
                [y,x,z],
            )
    else:
        raise ValueError(f"Method {method} is not a valid rotation method.")
    
    
    angle = np.deg2rad(angle)
    rotation_matrix = _get_rotation_matrix(angle, axes, dir=1)

    bbox_points_t = np.matmul(rotation_matrix, bbox_points.T)

    x_min, x_max = np.min(bbox_points_t[1]), np.max(bbox_points_t[1])
    y_min, y_max = np.min(bbox_points_t[0]), np.max(bbox_points_t[0])
    z_min, z_max = np.min(bbox_points_t[2]), np.max(bbox_points_t[2])
    bbox = x_min, y_min, z_min, x_max, y_max, z_max

    if crop_to_border:
        rows, cols, slices = _get_new_image_shape(rows, cols, slices, rotation_matrix)
    
    return list(map(lambda x: x + 0.5, normalize_bbox(bbox, rows, cols, slices)))


@angle_2pi_range
def keypoint_rotate(keypoint, angle: float, axes: str, crop_to_border: bool, rows: int, cols: int, slices: int, **params):
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

    p = np.array([[y,x,z]]) - in_center
    y,x,z = np.matmul(rotation_matrix, p.T).flatten() + out_center

    return x, y, z, a + (angle if axes=="xy" else 0), s




def _get_rotation_matrix(theta, axes, dir = 1):
    if axes == "xy":
        arr = np.array([
            [       np.cos(theta), dir * np.sin(theta), 0],
            [dir * -np.sin(theta),       np.cos(theta), 0],
            [                   0,                   0, 1],
        ],
        dtype= np.float32)
    
    elif axes == "yz":
        arr = np.array([
            [      np.cos(theta), 0, dir * -np.sin(theta)],
            [                  0, 1,                    0],
            [dir * np.sin(theta), 0,        np.cos(theta)],
        ],
        dtype= np.float32)

    elif axes == "xz":
        arr = np.array([
            [1,                   0,                    0],
            [0,       np.cos(theta), dir * -np.sin(theta)],
            [0, dir * np.sin(theta),        np.cos(theta)],
        ],
        dtype= np.float32)

    else:
        raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")
    
    return arr

# def _get_translation_matrix(x,y,z):
#     return np.array([
#         [1, 0, 0, y],
#         [0, 1, 0, x],
#         [0, 0, 1, z],
#         [0, 0, 0, 1],
#     ],
#     dtype= np.float32)

def _get_scale_matrix(dx,dy,dz):
    return np.array([
        [dy,  0,  0],
        [ 0, dx,  0],
        [ 0,  0, dz],
    ],
    dtype= np.float32)


def _get_shear_matrix(mode, sx = None, sy = None, sz = None):

    if sum(list(map(lambda x: x != None, [sx,sy,sz]))) != 2:
        raise ValueError("Expected two of (sx,sy,sz) to be non-null arguments. Got sx={}, sy={}, sz={}".format(sx,sy,sz))
    
    if mode == "xy":
        assert sx != None and sy != None
        
    elif mode == "xz":
        assert sx != None and sz != None

    elif mode == "yz":
        assert sy != None and sz != None
    
    else:
        raise ValueError("Parameter axes must be one of {'xy','yz','xz'}")

@preserve_channel_dim
def shift_scale_rotate(
    img, angle, scale, dx, dy, dz, axes = "xy", crop_to_border = False, interpolation=INTER_LINEAR, border_mode="constant", value=0
):
    out_shape = in_shape =  img.shape[:3]
    height, width, depth = out_shape
    in_center = _get_image_center(in_shape)
    out_center = _get_image_center(out_shape)
    
    
    scale = (scale,)*3
    angle = np.deg2rad(angle)

    rotation_matrix = _get_rotation_matrix(angle, axes, dir=-1)
    scale_matrix = _get_scale_matrix(*scale)

    if crop_to_border:
        out_shape = _get_new_image_shape(height, width, depth, rotation_matrix, *scale)
        out_center = _get_image_center(out_shape)

    matrix = np.linalg.inv(np.matmul(rotation_matrix, scale_matrix))
    offset = in_center - np.dot(matrix, out_center)
    shift = np.array([dy,dx,dz]) * np.array(out_shape)

    warp_affine_fn = _maybe_process_by_channel(
        ndimage.affine_transform, matrix= matrix, offset = offset, order=interpolation, output_shape=out_shape, mode=border_mode, cval=value
    )
    translate_fn = _maybe_process_by_channel(
        ndimage.shift, shift=shift,  order=interpolation, mode=border_mode, cval=value
    )
    return translate_fn(warp_affine_fn(img))


@angle_2pi_range
def keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, dz, axes = "xy", crop_to_border = False, rows=0, cols=0, slices=0,**params):
    
    out_shape = height, width, depth = rows, cols, slices
    in_center = np.array(out_shape) / 2 #_get_image_center((rows, cols, slices))
    out_center = in_center.copy()
    scale = (scale,)*3
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
    shift = np.array([dy,dx,dz]) * np.array(out_shape)

    x, y, z, a, s = keypoint[:5]

    p = np.array([[y,x,z]]) - in_center
    y,x,z = np.matmul(matrix, p.T).flatten() + out_center + shift

    return x, y, z, a + (angle if axes=="xy" else 0), s * scale[0]


def bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, dz, axes="xy", crop_to_border=False, rotate_method="largest_box", rows=0, cols=0, slices=0, **kwargs):  # skipcq: PYL-W0613
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

    x_min, y_min, z_min, x_max, y_max, z_max = denormalize_bbox(list(map(lambda x: x - 0.5, bbox[:6])), rows, cols, slices)

    if rotate_method == "largest_box":
        bbox_points = np.array(
            [
                [y_min, x_min,  z_min],
                [y_min, x_min,  z_max],
                [y_max, x_min,  z_min],
                [y_max, x_min,  z_max],
                [y_min, x_max,  z_min],
                [y_min, x_max,  z_max],
                [y_max, x_max,  z_min],
                [y_max, x_max,  z_max]
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
                [y,x,z],
            )
    else:
        raise ValueError(f"Method {rotate_method} is not a valid rotation method.")
    
    
    angle = np.deg2rad(angle)
    scale = (scale,)*3
    rotation_matrix = _get_rotation_matrix(angle, axes, dir=1)
    scale_matrix = _get_scale_matrix(*scale)
    shift = np.array([dx,dy,dz,dx,dy,dz])

    if crop_to_border:
        rows, cols, slices = _get_new_image_shape(rows, cols, slices, rotation_matrix, *scale)

    matrix = np.matmul(rotation_matrix, scale_matrix)

    bbox_points_t = np.matmul(matrix, bbox_points.T)

    x_min, x_max = np.min(bbox_points_t[1]), np.max(bbox_points_t[1])
    y_min, y_max = np.min(bbox_points_t[0]), np.max(bbox_points_t[0])
    z_min, z_max = np.min(bbox_points_t[2]), np.max(bbox_points_t[2])
    bbox = x_min, y_min, z_min, x_max, y_max, z_max
    
    # normalize points to assumed [0,1] range then apply translation
    return [p + s for s,p in zip(shift, map(lambda x: x + 0.5, normalize_bbox(bbox, rows, cols, slices)))]


# @preserve_shape
# def elastic_transform(
#     img: np.ndarray,
#     alpha: float,
#     sigma: float,
#     alpha_affine: float,
#     interpolation: int = cv2.INTER_LINEAR,
#     border_mode: int = cv2.BORDER_REFLECT_101,
#     value: Optional[ImageColorType] = None,
#     random_state: Optional[np.random.RandomState] = None,
#     approximate: bool = False,
#     same_dxdy: bool = False,
# ):
#     """Elastic deformation of images as described in [Simard2003]_ (with modifications).
#     Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#          Convolutional Neural Networks applied to Visual Document Analysis", in
#          Proc. of the International Conference on Document Analysis and
#          Recognition, 2003.
#     """
#     height, width = img.shape[:2]

#     # Random affine
#     center_square = np.array((height, width), dtype=np.float32) // 2
#     square_size = min((height, width)) // 3
#     alpha = float(alpha)
#     sigma = float(sigma)
#     alpha_affine = float(alpha_affine)

#     pts1 = np.array(
#         [
#             center_square + square_size,
#             [center_square[0] + square_size, center_square[1] - square_size],
#             center_square - square_size,
#         ],
#         dtype=np.float32,
#     )
#     pts2 = pts1 + random_utils.uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
#         np.float32
#     )
#     matrix = cv2.getAffineTransform(pts1, pts2)

#     warp_fn = _maybe_process_in_chunks(
#         cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
#     )
#     img = warp_fn(img)

#     if approximate:
#         # Approximate computation smooth displacement map with a large enough kernel.
#         # On large images (512+) this is approximately 2X times faster
#         dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
#         cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
#         dx *= alpha
#         if same_dxdy:
#             # Speed up even more
#             dy = dx
#         else:
#             dy = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
#             cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
#             dy *= alpha
#     else:
#         dx = np.float32(
#             ndimage.gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
#         )
#         if same_dxdy:
#             # Speed up
#             dy = dx
#         else:
#             dy = np.float32(
#                 ndimage.gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
#             )

#     x, y = np.meshgrid(np.arange(width), np.arange(height))

#     map_x = np.float32(x + dx)
#     map_y = np.float32(y + dy)

#     remap_fn = _maybe_process_in_chunks(
#         cv2.remap, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value
#     )
#     return remap_fn(img)

def _resize(img, dsize, interpolation):
    img_height, img_width, img_depth = img.shape[:3]
    dst_height, dst_width, dst_depth = dsize

    scale_y = dst_height / img_height
    scale_x = dst_width / img_width
    scale_z = dst_depth / img_depth

    return ndimage.zoom(img, zoom = (scale_y,scale_x,scale_z), order=interpolation)


@preserve_channel_dim
def resize(img, height, width, depth, interpolation=INTER_LINEAR):
    img_height, img_width, img_depth = img.shape[:3]
    if height == img_height and width == img_width and depth == img_depth:
        return img
    resize_fn = _maybe_process_by_channel(_resize, dsize=(height, width, depth), interpolation=interpolation)
    return resize_fn(img)


@preserve_channel_dim
def scale(img: np.ndarray, scale: Union[float, Tuple[float]], interpolation: int = INTER_LINEAR) -> np.ndarray:
    
    scale_fn = _maybe_process_by_channel(ndimage.zoom, zoom = scale, order=interpolation)
    return scale_fn(img)



def keypoint_scale(keypoint: KeypointInternalType, scale_x: float, scale_y: float, scale_z: float) -> KeypointInternalType:
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
    return x * scale_x, y * scale_y, z * scale_z, angle, scale * max((scale_x, scale_y, scale_z))


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
    return _func_max_size(img, max_size, interpolation, max)


@preserve_channel_dim
def smallest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    return _func_max_size(img, max_size, interpolation, min)


# @preserve_channel_dim
# def perspective(
#     img: np.ndarray,
#     matrix: np.ndarray,
#     max_width: int,
#     max_height: int,
#     border_val: Union[int, float, List[int], List[float], np.ndarray],
#     border_mode: int,
#     keep_size: bool,
#     interpolation: int,
# ):
#     h, w = img.shape[:2]
#     perspective_func = _maybe_process_in_chunks(
#         cv2.warpPerspective,
#         M=matrix,
#         dsize=(max_width, max_height),
#         borderMode=border_mode,
#         borderValue=border_val,
#         flags=interpolation,
#     )
#     warped = perspective_func(img)

#     if keep_size:
#         return resize(warped, h, w, interpolation=interpolation)

#     return warped


# def perspective_bbox(
#     bbox: BoxInternalType,
#     height: int,
#     width: int,
#     matrix: np.ndarray,
#     max_width: int,
#     max_height: int,
#     keep_size: bool,
# ) -> BoxInternalType:
#     x1, y1, x2, y2 = denormalize_bbox(bbox, height, width)[:4]

#     points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

#     x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
#     for pt in points:
#         pt = perspective_keypoint(pt.tolist() + [0, 0], height, width, matrix, max_width, max_height, keep_size)
#         x, y = pt[:2]
#         x1 = min(x1, x)
#         x2 = max(x2, x)
#         y1 = min(y1, y)
#         y2 = max(y2, y)

#     return normalize_bbox((x1, y1, x2, y2), height if keep_size else max_height, width if keep_size else max_width)


# def rotation2DMatrixToEulerAngles(matrix: np.ndarray, y_up: bool = False) -> float:
#     """
#     Args:
#         matrix (np.ndarray): Rotation matrix
#         y_up (bool): is Y axis looks up or down
#     """
#     if y_up:
#         return np.arctan2(matrix[1, 0], matrix[0, 0])
#     return np.arctan2(-matrix[1, 0], matrix[0, 0])


# @angle_2pi_range
# def perspective_keypoint(
#     keypoint: KeypointInternalType,
#     height: int,
#     width: int,
#     matrix: np.ndarray,
#     max_width: int,
#     max_height: int,
#     keep_size: bool,
# ) -> KeypointInternalType:
#     x, y, angle, scale = keypoint

#     keypoint_vector = np.array([x, y], dtype=np.float32).reshape([1, 1, 2])

#     x, y = cv2.perspectiveTransform(keypoint_vector, matrix)[0, 0]
#     angle += rotation2DMatrixToEulerAngles(matrix[:2, :2], y_up=True)

#     scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
#     scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
#     scale *= max(scale_x, scale_y)

#     if keep_size:
#         scale_x = width / max_width
#         scale_y = height / max_height
#         return keypoint_scale((x, y, angle, scale), scale_x, scale_y)

#     return x, y, angle, scale


# def _is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
#     return np.allclose(matrix.params, np.eye(3, dtype=np.float32))


# @preserve_channel_dim
# def warp_affine(
#     image: np.ndarray,
#     matrix: skimage.transform.ProjectiveTransform,
#     interpolation: int,
#     cval: Union[int, float, Sequence[int], Sequence[float]],
#     mode: int,
#     output_shape: Sequence[int],
# ) -> np.ndarray:
#     if _is_identity_matrix(matrix):
#         return image

#     dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))
#     warp_fn = _maybe_process_in_chunks(
#         cv2.warpAffine, M=matrix.params[:2], dsize=dsize, flags=interpolation, borderMode=mode, borderValue=cval
#     )
#     tmp = warp_fn(image)
#     return tmp


# @angle_2pi_range
# def keypoint_affine(
#     keypoint: KeypointInternalType,
#     matrix: skimage.transform.ProjectiveTransform,
#     scale: dict,
# ) -> KeypointInternalType:
#     if _is_identity_matrix(matrix):
#         return keypoint

#     x, y, a, s = keypoint[:4]
#     x, y = cv2.transform(np.array([[[x, y]]]), matrix.params[:2]).squeeze()
#     a += rotation2DMatrixToEulerAngles(matrix.params[:2])
#     s *= np.max([scale["x"], scale["y"]])
#     return x, y, a, s


# def bbox_affine(
#     bbox: BoxInternalType,
#     matrix: skimage.transform.ProjectiveTransform,
#     rotate_method: str,
#     rows: int,
#     cols: int,
#     output_shape: Sequence[int],
# ) -> BoxInternalType:
#     if _is_identity_matrix(matrix):
#         return bbox
#     x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
#     if rotate_method == "largest_box":
#         points = np.array(
#             [
#                 [x_min, y_min],
#                 [x_max, y_min],
#                 [x_max, y_max],
#                 [x_min, y_max],
#             ]
#         )
#     elif rotate_method == "ellipse":
#         w = (x_max - x_min) / 2
#         h = (y_max - y_min) / 2
#         data = np.arange(0, 360, dtype=np.float32)
#         x = w * np.sin(np.radians(data)) + (w + x_min - 0.5)
#         y = h * np.cos(np.radians(data)) + (h + y_min - 0.5)
#         points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
#     else:
#         raise ValueError(f"Method {rotate_method} is not a valid rotation method.")
#     points = skimage.transform.matrix_transform(points, matrix.params)
#     x_min = np.min(points[:, 0])
#     x_max = np.max(points[:, 0])
#     y_min = np.min(points[:, 1])
#     y_max = np.max(points[:, 1])

#     return normalize_bbox((x_min, y_min, x_max, y_max), output_shape[0], output_shape[1])


# @preserve_channel_dim
# def safe_rotate(
#     img: np.ndarray,
#     matrix: np.ndarray,
#     interpolation: int,
#     value: FillValueType = None,
#     border_mode: int = cv2.BORDER_REFLECT_101,
# ) -> np.ndarray:
#     h, w = img.shape[:2]
#     warp_fn = _maybe_process_in_chunks(
#         cv2.warpAffine,
#         M=matrix,
#         dsize=(w, h),
#         flags=interpolation,
#         borderMode=border_mode,
#         borderValue=value,
#     )
#     return warp_fn(img)


# def bbox_safe_rotate(bbox: BoxInternalType, matrix: np.ndarray, cols: int, rows: int) -> BoxInternalType:
#     x1, y1, x2, y2 = denormalize_bbox(bbox, rows, cols)[:4]
#     points = np.array(
#         [
#             [x1, y1, 1],
#             [x2, y1, 1],
#             [x2, y2, 1],
#             [x1, y2, 1],
#         ]
#     )
#     points = points @ matrix.T
#     x1 = points[:, 0].min()
#     x2 = points[:, 0].max()
#     y1 = points[:, 1].min()
#     y2 = points[:, 1].max()

#     def fix_point(pt1: float, pt2: float, max_val: float) -> Tuple[float, float]:
#         # In my opinion, these errors should be very low, around 1-2 pixels.
#         if pt1 < 0:
#             return 0, pt2 + pt1
#         if pt2 > max_val:
#             return pt1 - (pt2 - max_val), max_val
#         return pt1, pt2

#     x1, x2 = fix_point(x1, x2, cols)
#     y1, y2 = fix_point(y1, y2, rows)

#     return normalize_bbox((x1, y1, x2, y2), rows, cols)


# def keypoint_safe_rotate(
#     keypoint: KeypointInternalType,
#     matrix: np.ndarray,
#     angle: float,
#     scale_x: float,
#     scale_y: float,
#     cols: int,
#     rows: int,
# ) -> KeypointInternalType:
#     x, y, a, s = keypoint[:4]
#     point = np.array([[x, y, 1]])
#     x, y = (point @ matrix.T)[0]

#     # To avoid problems with float errors
#     x = np.clip(x, 0, cols - 1)
#     y = np.clip(y, 0, rows - 1)

#     a += angle
#     s *= max(scale_x, scale_y)
#     return x, y, a, s


# @clipped
# def piecewise_affine(
#     img: np.ndarray,
#     matrix: skimage.transform.PiecewiseAffineTransform,
#     interpolation: int,
#     mode: str,
#     cval: float,
# ) -> np.ndarray:
#     return skimage.transform.warp(
#         img, matrix, order=interpolation, mode=mode, cval=cval, preserve_range=True, output_shape=img.shape
#     )


# def to_distance_maps(
#     keypoints: Sequence[Tuple[float, float]], height: int, width: int, inverted: bool = False
# ) -> np.ndarray:
#     """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.

#     The ``n``-th distance map contains at every location ``(y, x)`` the
#     euclidean distance to the ``n``-th keypoint.

#     This function can be used as a helper when augmenting keypoints with a
#     method that only supports the augmentation of images.

#     Args:
#         keypoint: keypoint coordinates
#         height: image height
#         width: image width
#         inverted (bool): If ``True``, inverted distance maps are returned where each
#             distance value d is replaced by ``d/(d+1)``, i.e. the distance
#             maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
#             exactly the position of the respective keypoint.

#     Returns:
#         (H, W, N) ndarray
#             A ``float32`` array containing ``N`` distance maps for ``N``
#             keypoints. Each location ``(y, x, n)`` in the array denotes the
#             euclidean distance at ``(y, x)`` to the ``n``-th keypoint.
#             If `inverted` is ``True``, the distance ``d`` is replaced
#             by ``d/(d+1)``. The height and width of the array match the
#             height and width in ``KeypointsOnImage.shape``.
#     """
#     distance_maps = np.zeros((height, width, len(keypoints)), dtype=np.float32)

#     yy = np.arange(0, height)
#     xx = np.arange(0, width)
#     grid_xx, grid_yy = np.meshgrid(xx, yy)

#     for i, (x, y) in enumerate(keypoints):
#         distance_maps[:, :, i] = (grid_xx - x) ** 2 + (grid_yy - y) ** 2

#     distance_maps = np.sqrt(distance_maps)
#     if inverted:
#         return 1 / (distance_maps + 1)
#     return distance_maps


# def from_distance_maps(
#     distance_maps: np.ndarray,
#     inverted: bool,
#     if_not_found_coords: Optional[Union[Sequence[int], dict]],
#     threshold: Optional[float] = None,
# ) -> List[Tuple[float, float]]:
#     """Convert outputs of ``to_distance_maps()`` to ``KeypointsOnImage``.
#     This is the inverse of `to_distance_maps`.

#     Args:
#         distance_maps (np.ndarray): The distance maps. ``N`` is the number of keypoints.
#         inverted (bool): Whether the given distance maps were generated in inverted mode
#             (i.e. :func:`KeypointsOnImage.to_distance_maps` was called with ``inverted=True``) or in non-inverted mode.
#         if_not_found_coords (tuple, list, dict or None, optional):
#             Coordinates to use for keypoints that cannot be found in `distance_maps`.

#             * If this is a ``list``/``tuple``, it must contain two ``int`` values.
#             * If it is a ``dict``, it must contain the keys ``x`` and ``y`` with each containing one ``int`` value.
#             * If this is ``None``, then the keypoint will not be added.
#         threshold (float): The search for keypoints works by searching for the
#             argmin (non-inverted) or argmax (inverted) in each channel. This
#             parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
#             as a keypoint. Use ``None`` to use no min/max.
#         nb_channels (None, int): Number of channels of the image on which the keypoints are placed.
#             Some keypoint augmenters require that information. If set to ``None``, the keypoint's shape will be set
#             to ``(height, width)``, otherwise ``(height, width, nb_channels)``.
#     """
#     if distance_maps.ndim != 3:
#         raise ValueError(
#             f"Expected three-dimensional input, "
#             f"got {distance_maps.ndim} dimensions and shape {distance_maps.shape}."
#         )
#     height, width, nb_keypoints = distance_maps.shape

#     drop_if_not_found = False
#     if if_not_found_coords is None:
#         drop_if_not_found = True
#         if_not_found_x = -1
#         if_not_found_y = -1
#     elif isinstance(if_not_found_coords, (tuple, list)):
#         if len(if_not_found_coords) != 2:
#             raise ValueError(
#                 f"Expected tuple/list 'if_not_found_coords' to contain exactly two entries, "
#                 f"got {len(if_not_found_coords)}."
#             )
#         if_not_found_x = if_not_found_coords[0]
#         if_not_found_y = if_not_found_coords[1]
#     elif isinstance(if_not_found_coords, dict):
#         if_not_found_x = if_not_found_coords["x"]
#         if_not_found_y = if_not_found_coords["y"]
#     else:
#         raise ValueError(
#             f"Expected if_not_found_coords to be None or tuple or list or dict, got {type(if_not_found_coords)}."
#         )

#     keypoints = []
#     for i in range(nb_keypoints):
#         if inverted:
#             hitidx_flat = np.argmax(distance_maps[..., i])
#         else:
#             hitidx_flat = np.argmin(distance_maps[..., i])
#         hitidx_ndim = np.unravel_index(hitidx_flat, (height, width))
#         if not inverted and threshold is not None:
#             found = distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] < threshold
#         elif inverted and threshold is not None:
#             found = distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] >= threshold
#         else:
#             found = True
#         if found:
#             keypoints.append((float(hitidx_ndim[1]), float(hitidx_ndim[0])))
#         else:
#             if not drop_if_not_found:
#                 keypoints.append((if_not_found_x, if_not_found_y))

#     return keypoints


# def keypoint_piecewise_affine(
#     keypoint: KeypointInternalType,
#     matrix: skimage.transform.PiecewiseAffineTransform,
#     h: int,
#     w: int,
#     keypoints_threshold: float,
# ) -> KeypointInternalType:
#     x, y, a, s = keypoint[:4]
#     dist_maps = to_distance_maps([(x, y)], h, w, True)
#     dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
#     x, y = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)[0]
#     return x, y, a, s


# def bbox_piecewise_affine(
#     bbox: BoxInternalType,
#     matrix: skimage.transform.PiecewiseAffineTransform,
#     h: int,
#     w: int,
#     keypoints_threshold: float,
# ) -> BoxInternalType:
#     x1, y1, x2, y2 = denormalize_bbox(bbox, h, w)[:4]
#     keypoints = [
#         (x1, y1),
#         (x2, y1),
#         (x2, y2),
#         (x1, y2),
#     ]
#     dist_maps = to_distance_maps(keypoints, h, w, True)
#     dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
#     keypoints = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)
#     keypoints = [i for i in keypoints if 0 <= i[0] < w and 0 <= i[1] < h]
#     keypoints_arr = np.array(keypoints)
#     x1 = keypoints_arr[:, 0].min()
#     y1 = keypoints_arr[:, 1].min()
#     x2 = keypoints_arr[:, 0].max()
#     y2 = keypoints_arr[:, 1].max()
#     return normalize_bbox((x1, y1, x2, y2), h, w)


def vflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ...])

def hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1, ...])

def zflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, :, ::-1, ...])

def hflip_cv2(img: np.ndarray) -> np.ndarray:
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
        raise ValueError("Invalid d value {}. Valid values are -1, 0, 1, and 2".format(d))
    return img


def transpose(img: np.ndarray) -> np.ndarray:
    return img.transpose(1, 0, 2, 3) if len(img.shape) > 3 else img.transpose(1, 0, 2)


def rot90(img: np.ndarray, factor: int, axes: Tuple = (0,1)) -> np.ndarray:
    img = np.rot90(img, factor, axes)
    return np.ascontiguousarray(img)


def bbox_vflip(bbox: BoxInternalType, rows: int, cols: int, slices: int) -> BoxInternalType:  # skipcq: PYL-W0613
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


def bbox_hflip(bbox: BoxInternalType, rows: int, cols: int, slices: int) -> BoxInternalType:  # skipcq: PYL-W0613
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

def bbox_zflip(bbox: BoxInternalType, rows: int, cols: int, slices: int) -> BoxInternalType:  # skipcq: PYL-W0613
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


def bbox_flip(bbox: BoxInternalType, d: int, rows: int, cols: int, slices: int) -> BoxInternalType:
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
        raise ValueError("Invalid d value {}. Valid values are -1, 0, 1, and 2".format(d))
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
    if axis not in {0,1}:
        raise ValueError("Parameter axes must be one of {0,1}")
    if axis == 0:
        bbox = (y_min, x_min, z_min, y_max, x_max, z_max)
    if axis == 1:
        bbox = (1 - y_max, 1 - x_max, z_min, 1 - y_min, 1 - x_min, z_max)
    return bbox


@angle_2pi_range
def keypoint_vflip(keypoint: KeypointInternalType, rows: int, cols: int, slices: int) -> KeypointInternalType:
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
def keypoint_hflip(keypoint: KeypointInternalType, rows: int, cols: int, slices: int) -> KeypointInternalType:
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
def keypoint_zflip(keypoint: KeypointInternalType, rows: int, cols: int, slices: int) -> KeypointInternalType:
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
    return x, y, (slices-1) - z, angle, scale


def keypoint_flip(keypoint: KeypointInternalType, d: int, rows: int, cols: int, slices: int) -> KeypointInternalType:
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
    value: Union[float,int] = 0,
) -> np.ndarray:
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

    img = pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, d_pad_close, d_pad_far, border_mode, value)

    if img.shape[:3] != (max(min_height, height), max(min_width, width), max(min_depth, depth)):
        raise RuntimeError(
            "Invalid result shape. Got: {}. Expected: {}".format(
                img.shape[:3], (max(min_height, height), max(min_width, width), max(min_depth, depth))
            )
        )

    return img


def _pad(
    img: np.ndarray,
    pad_width: Tuple[Tuple,Tuple,Tuple],
    border_mode: str = 'constant',
    value: Union[float,int] = 0
) -> np.ndarray:
    
    return np.pad(img, pad_width=pad_width, mode=SCIPY_MODE_TO_NUMPY_MODE[border_mode], constant_values = value)



@preserve_channel_dim
def pad_with_params(
    img: np.ndarray,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    d_pad_front: int,
    d_pad_back: int,
    border_mode: str = 'constant',
    value: Union[float,int] = None,
) -> np.ndarray:
    pad_fn = _maybe_process_by_channel(
        _pad,
        pad_width=((h_pad_top, h_pad_bottom),(w_pad_left, w_pad_right),(d_pad_front, d_pad_back)),
        border_mode=border_mode,
        value=value)
    return pad_fn(img)


# @preserve_shape
# def optical_distortion(
#     img: np.ndarray,
#     k: int = 0,
#     dx: int = 0,
#     dy: int = 0,
#     interpolation: int = cv2.INTER_LINEAR,
#     border_mode: int = cv2.BORDER_REFLECT_101,
#     value: Optional[ImageColorType] = None,
# ) -> np.ndarray:
#     """Barrel / pincushion distortion. Unconventional augment.

#     Reference:
#         |  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
#         |  https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
#         |  https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
#         |  http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
#     """
#     height, width = img.shape[:2]

#     fx = width
#     fy = height

#     cx = width * 0.5 + dx
#     cy = height * 0.5 + dy

#     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

#     distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
#     map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
#     return cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=value)


# @preserve_shape
# def grid_distortion(
#     img: np.ndarray,
#     num_steps: int = 10,
#     xsteps: Tuple = (),
#     ysteps: Tuple = (),
#     interpolation: int = cv2.INTER_LINEAR,
#     border_mode: int = cv2.BORDER_REFLECT_101,
#     value: Optional[ImageColorType] = None,
# ) -> np.ndarray:
#     """Perform a grid distortion of an input image.

#     Reference:
#         http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
#     """
#     height, width = img.shape[:2]

#     x_step = width // num_steps
#     xx = np.zeros(width, np.float32)
#     prev = 0
#     for idx in range(num_steps + 1):
#         x = idx * x_step
#         start = int(x)
#         end = int(x) + x_step
#         if end > width:
#             end = width
#             cur = width
#         else:
#             cur = prev + x_step * xsteps[idx]

#         xx[start:end] = np.linspace(prev, cur, end - start)
#         prev = cur

#     y_step = height // num_steps
#     yy = np.zeros(height, np.float32)
#     prev = 0
#     for idx in range(num_steps + 1):
#         y = idx * y_step
#         start = int(y)
#         end = int(y) + y_step
#         if end > height:
#             end = height
#             cur = height
#         else:
#             cur = prev + y_step * ysteps[idx]

#         yy[start:end] = np.linspace(prev, cur, end - start)
#         prev = cur

#     map_x, map_y = np.meshgrid(xx, yy)
#     map_x = map_x.astype(np.float32)
#     map_y = map_y.astype(np.float32)

#     remap_fn = _maybe_process_in_chunks(
#         cv2.remap,
#         map1=map_x,
#         map2=map_y,
#         interpolation=interpolation,
#         borderMode=border_mode,
#         borderValue=value,
#     )
#     return remap_fn(img)


# @preserve_shape
# def elastic_transform_approx(
#     img: np.ndarray,
#     alpha: float,
#     sigma: float,
#     alpha_affine: float,
#     interpolation: int = cv2.INTER_LINEAR,
#     border_mode: int = cv2.BORDER_REFLECT_101,
#     value: Optional[ImageColorType] = None,
#     random_state: Optional[np.random.RandomState] = None,
# ) -> np.ndarray:
#     """Elastic deformation of images as described in [Simard2003]_ (with modifications for speed).
#     Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

#     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
#          Convolutional Neural Networks applied to Visual Document Analysis", in
#          Proc. of the International Conference on Document Analysis and
#          Recognition, 2003.
#     """
#     height, width = img.shape[:2]

#     # Random affine
#     center_square = np.array((height, width), dtype=np.float32) // 2
#     square_size = min((height, width)) // 3
#     alpha = float(alpha)
#     sigma = float(sigma)
#     alpha_affine = float(alpha_affine)

#     pts1 = np.array(
#         [
#             center_square + square_size,
#             [center_square[0] + square_size, center_square[1] - square_size],
#             center_square - square_size,
#         ],
#         dtype=np.float32,
#     )
#     pts2 = pts1 + random_utils.uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
#         np.float32
#     )
#     matrix = cv2.getAffineTransform(pts1, pts2)

#     warp_fn = _maybe_process_in_chunks(
#         cv2.warpAffine,
#         M=matrix,
#         dsize=(width, height),
#         flags=interpolation,
#         borderMode=border_mode,
#         borderValue=value,
#     )
#     img = warp_fn(img)

#     dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
#     cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
#     dx *= alpha

#     dy = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
#     cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
#     dy *= alpha

#     x, y = np.meshgrid(np.arange(width), np.arange(height))

#     map_x = np.float32(x + dx)
#     map_y = np.float32(y + dy)

#     remap_fn = _maybe_process_in_chunks(
#         cv2.remap,
#         map1=map_x,
#         map2=map_y,
#         interpolation=interpolation,
#         borderMode=border_mode,
#         borderValue=value,
#     )
#     return remap_fn(img)
