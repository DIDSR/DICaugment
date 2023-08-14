from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from dicaugment.augmentations.utils import (
    _maybe_process_in_chunks,
    preserve_channel_dim,
)

from ...core.bbox_utils import denormalize_bbox, normalize_bbox
from ...core.transforms_interface import BoxInternalType, KeypointInternalType
from ..geometric import functional as FGeometric

__all__ = [
    "get_random_crop_coords",
    "random_crop",
    "crop_bbox_by_coords",
    "bbox_random_crop",
    "crop_keypoint_by_coords",
    "keypoint_random_crop",
    "get_center_crop_coords",
    "center_crop",
    "bbox_center_crop",
    "keypoint_center_crop",
    "crop",
    "bbox_crop",
    "clamping_crop",
    "crop_and_pad",
    "crop_and_pad_bbox",
    "crop_and_pad_keypoint",
]


def get_random_crop_coords(height: int, width: int, depth: int, crop_height: int, crop_width: int, crop_depth: int, h_start: float, w_start: float, d_start: float):
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width
    z1 = int((depth - crop_depth + 1) * d_start)
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def random_crop(img: np.ndarray, crop_height: int, crop_width: int, crop_depth: int, h_start: float, w_start: float, d_start: float):
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}, {crop_depth}) is "
            "larger than the image size ({height}, {width}, {depth})".format(
                crop_height=crop_height, crop_width=crop_width, crop_depth=crop_depth, height=height, width=width, depth=depth
            )
        )
    x1, y1, z1, x2, y2, z2 = get_random_crop_coords(height, width, depth, crop_height, crop_width, crop_depth, h_start, w_start, d_start)
    img = img[y1:y2, x1:x2, z1:z2]
    return img


def crop_bbox_by_coords(
    bbox: BoxInternalType,
    crop_coords: Tuple[int, int, int, int, int, int],
    crop_height: int,
    crop_width: int,
    crop_depth: int,
    rows: int,
    cols: int,
    slices: int
):
    """Crop a bounding box using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        bbox (tuple): A cropped box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        crop_coords (tuple): Crop coordinates `(x1, y1, z1, x2, y2, z2)`.
        crop_height (int):
        crop_width (int):
        crop_depth (int):
        rows (int): Image rows.
        cols (int): Image cols.
        slices (int): Image slices.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max, z_min, z_max)`.

    """
    bbox = denormalize_bbox(bbox, rows, cols, slices)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    x1, y1, z1, x2, y2, z2 = crop_coords
    cropped_bbox = x_min - x1, y_min - y1, z_min - z1, x_max - x1, y_max - y1, z_max - z1
    return normalize_bbox(cropped_bbox, crop_height, crop_width, crop_depth)


def bbox_random_crop(
    bbox: BoxInternalType, crop_height: int, crop_width: int, crop_depth: int, h_start: float, w_start: float, d_start: float, rows: int, cols: int, slices: int
):
    crop_coords = get_random_crop_coords(rows, cols, slices, crop_height, crop_width, crop_depth, h_start, w_start, d_start)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, crop_depth, rows, cols, slices)


def crop_keypoint_by_coords(
    keypoint: KeypointInternalType, crop_coords: Tuple[int, int, int, int, int, int]
):  # skipcq: PYL-W0613
    """Crop a keypoint using the provided coordinates of closest-top-left and furthest-bottom-right corners in pixels and the
    required height, width, and depth of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, z, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, y1, z1, x2, y2, z2)`.

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    """
    x, y, z, angle, scale = keypoint[:5]
    x1, y1, z1, x2, y2, z2= crop_coords
    return x - x1, y - y1, z - z1, angle, scale


def keypoint_random_crop(
    keypoint: KeypointInternalType,
    crop_height: int,
    crop_width: int,
    crop_depth: int,
    h_start: float,
    w_start: float,
    d_start: float,
    rows: int,
    cols: int,
    slices: int
):
    """Keypoint random crop.

    Args:
        keypoint: (tuple): A keypoint `(x, y, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        crop_depth (int): Crop depth.
        h_start (int): Crop height start.
        w_start (int): Crop width start.
        d_start (int): Crop depth start.
        rows (int): Image height.
        cols (int): Image width.
        slices (int): Image depth

    Returns:
        A keypoint `(x, y, z, angle, scale)`.

    """
    crop_coords = get_random_crop_coords(rows, cols, slices, crop_height, crop_width, crop_depth, h_start, w_start, d_start)
    return crop_keypoint_by_coords(keypoint, crop_coords)


def get_center_crop_coords(height: int, width: int, depth: int, crop_height: int, crop_width: int, crop_depth: int):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    z1 = (depth - crop_depth) // 2
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def center_crop(img: np.ndarray, crop_height: int, crop_width: int, crop_depth: int):
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}, {crop_depth}) is "
            "larger than the image size ({height}, {width}, {depth})".format(
                crop_height=crop_height, crop_width=crop_width, crop_depth=crop_depth, height=height, width=width, depth=depth
            )
        )
    x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
    img = img[y1:y2, x1:x2, z1:z2]
    return img


def bbox_center_crop(bbox: BoxInternalType, crop_height: int, crop_width: int, crop_depth:int, rows: int, cols: int, slices: int):
    crop_coords = get_center_crop_coords(rows, cols, slices, crop_height, crop_width, crop_depth)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, crop_depth, rows, cols, slices)


def keypoint_center_crop(keypoint: KeypointInternalType, crop_height: int, crop_width: int, crop_depth:int, rows: int, cols: int, slices: int):
    """Keypoint center crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, z, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        crop_depth (int): Crop depth.
        rows (int): Image height.
        cols (int): Image width.
        slices (int): Image depths.

    Returns:
        tuple: A keypoint `(x, y, z, angle, scale)`.

    """
    crop_coords = get_center_crop_coords(rows, cols, slices, crop_height, crop_width, crop_depth)
    return crop_keypoint_by_coords(keypoint, crop_coords)


def crop(img: np.ndarray, x_min: int, y_min: int, z_min: int, x_max: int, y_max: int, z_max: int):
    height, width, depth = img.shape[:3]
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "Expected x_min < x_max, y_min < y_max, and z_min < z_max. Got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min}, x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height or z_min < 0 or z_max > depth:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes. Got "
            "(x_min = {x_min}, y_min = {y_min}, z_min = {z_min}, x_max = {x_max}, y_max = {y_max}, z_max = {z_max}, "
            "height = {height}, width = {width}, depth = {depth})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max, height=height, width=width, depth = depth
            )
        )

    return img[y_min:y_max, x_min:x_max, z_min:z_max]


def bbox_crop(bbox: BoxInternalType, x_min: int, y_min: int, z_min: int, x_max: int, y_max: int, z_max: int, rows: int, cols: int, slices: int):
    """Crop a bounding box.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        x_min (int): 
        y_min (int):
        z_min (int):
        x_max (int):
        y_max (int):
        z_max (int):
        rows (int): Image width.
        cols (int): Image height.
        slices (int): Image depth.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """
    crop_coords = x_min, y_min, z_min, x_max, y_max, z_max
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    crop_depth = z_max - z_min
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, crop_depth, rows, cols, slices)


def clamping_crop(img: np.ndarray, x_min: int, y_min: int, z_min: int, x_max: int, y_max: int, z_max: int):
    h, w, d = img.shape[:3]

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    z_min = max(z_min, 0)
    x_max = min(x_max, w - 1)
    y_max = min(y_max, h - 1)
    z_max = min(z_max, d - 1)

    return img[int(y_min) : int(y_max), int(x_min) : int(x_max), int(z_min) : int(z_max)]

@preserve_channel_dim
def crop_and_pad(
    img: np.ndarray,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    pad_value: Optional[float],
    rows: int,
    cols: int,
    slices: int,
    interpolation: int,
    pad_mode: int,
    keep_size: bool,
) -> np.ndarray:
    if crop_params is not None and any(i != 0 for i in crop_params):
        img = crop(img, *crop_params)
    if pad_params is not None and any(i != 0 for i in pad_params):
        img = FGeometric.pad_with_params(
            img, *pad_params[:6], border_mode=pad_mode, value=pad_value
        )

    if keep_size:
        img = FGeometric.resize(img, height = cols, width = rows, depth = slices, interpolation = interpolation )

    return img


def crop_and_pad_bbox(
    bbox: BoxInternalType,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows,
    cols,
    slices,
    result_rows,
    result_cols,
    result_slices,
) -> BoxInternalType:
    
    x1, y1, z1, x2, y2, z2 = denormalize_bbox(bbox, rows, cols, slices)[:6]

    if crop_params is not None:
        crop_x, _, crop_y, _, crop_z, _ = crop_params
        x1, y1, z1, x2, y2, z2 = x1 - crop_x, y1 - crop_y, z1 - crop_z, x2 - crop_x, y2 - crop_y, z2 - crop_z
    if pad_params is not None:
        top, bottom, left, right, close, far = pad_params
        x1, y1, z1, x2, y2, z2 = x1 + left, y1 + top, z1 + close, x2 + left, y2 + top, z2 + close

    return normalize_bbox((x1, y1, z1, x2, y2, z2), result_rows, result_cols, result_slices)


def crop_and_pad_keypoint(
    keypoint: KeypointInternalType,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows: int,
    cols: int,
    slices: int,
    result_rows: int,
    result_cols: int,
    result_slices: int,
    keep_size: bool,
) -> KeypointInternalType:
    x, y, z, angle, scale = keypoint[:5]

    if crop_params is not None:
        crop_x, _, crop_y, _, crop_z, _ = crop_params
        x, y, z = x - crop_x, y - crop_y, z - crop_z
    if pad_params is not None:
        top, bottom, left, right, close, far = pad_params
        x, y, z = x + left, y + top, z + close

    if keep_size and (result_cols != cols or result_rows != rows or result_cols != slices):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        scale_z = slices / result_slices
        return FGeometric.keypoint_scale((x, y, z, angle, scale), scale_x, scale_y, scale_z)

    return x, y, z, angle, scale
