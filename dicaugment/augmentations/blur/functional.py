from itertools import product
from math import ceil
from typing import Sequence, Union

import cv2
import numpy as np
from scipy import ndimage

from dicaugment.augmentations.functional import convolve
from dicaugment.augmentations.geometric.functional import scale
from dicaugment.augmentations.utils import (
    _maybe_process_in_chunks,
    _maybe_process_by_channel,
    clipped,
    preserve_shape,
)

__all__ = [
    "blur",
    "median_blur",
    "gaussian_blur",
]


@preserve_shape
def blur(
    img: np.ndarray,
    ksize: int,
    by_slice: bool = False,
    mode: str = "constant",
    cval: Union[float, int] = 0,
) -> np.ndarray:
    """Blur the input image using an mean kernel.

    Args:
        ksize (int): The kernel size for blurring the input image.
        by_slice (bool): Whether the kernel should be applied by slice or the image as a whole. If true, a 2D kernel is convolved along each slice of the image.
            Otherwise, a 3D kernel is used. Default: False
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            - `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            - `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            - `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            - `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            - `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        cval (int,float): The fill value when mode = `constant`. Default: 0

    """
    if by_slice:
        kernel = np.ones((ksize, ksize, 1), dtype=np.float32)
    else:
        kernel = np.ones((ksize,) * 3, dtype=np.float32)
    kernel /= np.sum(kernel)

    return convolve(img, kernel=kernel, mode=mode, cval=cval)


@preserve_shape
def median_blur(
    img: np.ndarray,
    ksize: int,
    by_slice: bool = False,
    mode: str = "constant",
    cval: Union[float, int] = 0,
) -> np.ndarray:
    """Blur the input image using median blue technique.

    Args:
        ksize (int): The kernel size for blurring the input image.
        by_slice (bool): Whether the kernel should be applied by slice or the image as a whole. If true, a 2D kernel is convolved along each slice of the image.
            Otherwise, a 3D kernel is used. Default: False
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            - `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            - `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            - `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            - `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            - `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        cval (int,float): The fill value when mode = `constant`. Default: 0

    """
    if by_slice:
        ksize = (ksize, ksize, 1)

    blur_fn = _maybe_process_by_channel(
        ndimage.median_filter, size=ksize, mode=mode, cval=cval
    )
    return blur_fn(img)


@preserve_shape
def gaussian_blur(
    img: np.ndarray,
    ksize: int,
    sigma: float = 0,
    by_slice: bool = False,
    mode: str = "constant",
    cval: Union[float, int] = 0,
) -> np.ndarray:
    """Blur the input image using a Gaussian kernel.

    Args:
        ksize (int): The kernel size for blurring the input image. If 0, then ksize is estimated as `round(sigma * 8) + 1`
        sigma (float): Gaussian kernel standard deviation. If 0, then sigma is estimated as `0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`
        by_slice (bool): Whether the kernel should be applied by slice or the image as a whole. If true, a 2D kernel is convolved along each slice of the image.
            Otherwise, a 3D kernel is used. Default: False
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            - `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            - `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            - `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            - `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            - `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        cval (int,float): The fill value when mode = `constant`. Default: 0

    """
    if ksize == 0:
        ksize = round(sigma * 8) + 1

    if sigma == 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    if by_slice:
        radius = ((ksize - 1) // 2, (ksize - 1) // 2, 0)
    else:
        radius = ((ksize - 1) // 2,) * 3

    blur_fn = _maybe_process_by_channel(
        ndimage.gaussian_filter, sigma=sigma, radius=radius, mode=mode, cval=cval
    )
    return blur_fn(img)
