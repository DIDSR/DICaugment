import random
import warnings
from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np

from dicaugment import random_utils
from dicaugment.augmentations import functional as FMain
from dicaugment.augmentations.blur import functional as F
from dicaugment.core.transforms_interface import (
    ImageOnlyTransform,
    ScaleFloatType,
    ScaleIntType,
    to_tuple,
)

__all__ = [
    "Blur",
    "GaussianBlur",
    "MedianBlur",
]


class Blur(ImageOnlyTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
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
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        by_slice: bool = False,
        mode: str = "constant",
        cval: Union[float, int] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.mode = mode
        self.by_slice = by_slice
        self.cval = cval

        if self.mode not in {"reflect", "constant", "nearest", "mirror", "wrap"}:
            raise ValueError(
                "Expected mode to be one of ('reflect', 'constant', 'nearest', 'mirror', 'wrap'), got {}".format(
                    self.mode
                )
            )

    def apply(self, img: np.ndarray, ksize: int = 3, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.blur(
            img, ksize, by_slice=self.by_slice, mode=self.mode, cval=self.cval
        )

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "ksize": int(
                random.choice(
                    list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2))
                )
            )
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("blur_limit", "by_slice", "mode", "cval")

class MedianBlur(Blur):
    """Blur the input image using a median filter with a random aperture linear size.

    Args:
        blur_limit (int): maximum aperture linear size for blurring the input image.
            Must be odd and in range [3, inf). Default: (3, 7).
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
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        by_slice: bool = False,
        mode: str = "constant",
        cval: Union[float, int] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(blur_limit, by_slice, mode, cval, always_apply, p)

        if self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError("MedianBlur supports only odd blur limits.")

    def apply(self, img: np.ndarray, ksize: int = 3, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.median_blur(
            img, ksize, by_slice=self.by_slice, mode=self.mode, cval=self.cval
        )


class GaussianBlur(ImageOnlyTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * 4 * 2) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
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
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0,
        by_slice: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
        mode: str = "constant",
        cval: Union[float, int] = 0,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 0)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)
        self.by_slice = by_slice
        self.mode = mode
        self.cval = cval

        if self.mode not in {"reflect", "constant", "nearest", "mirror", "wrap"}:
            raise ValueError(
                "Expected mode to be one of ('reflect', 'constant', 'nearest', 'mirror', 'wrap'), got {}".format(
                    self.mode
                )
            )

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            warnings.warn(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

    def apply(
        self, img: np.ndarray, ksize: int = 3, sigma: float = 0, **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.gaussian_blur(
            img,
            ksize,
            sigma=sigma,
            by_slice=self.by_slice,
            mode=self.mode,
            cval=self.cval,
        )

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("blur_limit", "sigma_limit", "by_slice", "mode", "cval")
