from __future__ import absolute_import, division

import math
import numbers
import random
import warnings
from enum import IntEnum
from types import LambdaType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import special
from scipy.ndimage import gaussian_filter

from dicaugment import random_utils
from dicaugment.augmentations.blur.functional import blur
from dicaugment.augmentations.utils import (
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
)

from ..core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    ScaleFloatType,
    BoxInternalType,
    KeypointInternalType,
    to_tuple,
    INTER_NEAREST,
    INTER_LINEAR,
    INTER_QUADRATIC,
    INTER_CUBIC,
    INTER_QUARTIC,
    INTER_QUINTIC,
)
from ..core.utils import format_args
from . import functional as F

__all__ = [
    "Normalize",
    "RandomGamma",
    "GaussNoise",
    "InvertImg",
    "ToFloat",
    "FromFloat",
    "RandomBrightnessContrast",
    "Equalize",
    "Posterize",
    "Downscale",
    "Sharpen",
    "UnsharpMask",
    "PixelDropout",
]


class Normalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean) / (std)`

    Args:
        mean (None, float, list of float): mean values along channel dimension. If None, mean is calculated per image at runtime.
        std  (None, float, list of float): std values along channel dimension. If None, std is calculated per image at runtime.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean: Union[None, float, Tuple[float]] = None,
        std: Union[None, float, Tuple[float]] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.normalize(image, self.mean, self.std)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("mean", "std")

class Posterize(ImageOnlyTransform):
    """Reduce the number of bits for each color channel.

    Args:
        num_bits ((int, int) or int, or list of ints [r, g, b], or list of ints [[r1, r2], [g1, g2], [b1, b2]]): number of high bits.
            If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, n] where n is the number of bits in the image dtype . Default: 8.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
    image

    Image types:
        uint8, uint16, int16, int32
    """

    def __init__(self, num_bits=8, always_apply=False, p=0.5):
        super(Posterize, self).__init__(always_apply, p)

        if isinstance(num_bits, (list, tuple)):
            if len(num_bits) == 3:
                self.num_bits = [to_tuple(i, 0) for i in num_bits]
            else:
                self.num_bits = to_tuple(num_bits, 0)
        else:
            self.num_bits = to_tuple(num_bits, num_bits)

    def apply(self, image, num_bits=1, **params):
        """Applies the transformation to the image"""
        return F.posterize(image, num_bits)

    def get_params(self):
        """Returns parameters needed for the `apply` methods"""
        if len(self.num_bits) > 2:
            return {"num_bits": [random.randint(i[0], i[1]) for i in self.num_bits]}
        return {"num_bits": random.randint(self.num_bits[0], self.num_bits[1])}

    def get_transform_init_args_names(self):
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("num_bits",)


class Equalize(ImageOnlyTransform):
    """Equalize the image histogram. For multi-channel images, each channel is processed individually

    Args:
        range (int, list of int): Histogram range.
            If int, then range is defined as [0, range].
            If None, the range is calculated as [0, max(img)].
            Default: None
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis.
            Function signature must include `image` argument.
        mask_params (list of str): Params for mask function.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, int16
    """

    def __init__(
        self,
        range: Union[int, Tuple[int, int]] = None,
        mask: Union[np.ndarray, callable] = None,
        mask_params: Sequence[str] = (),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(Equalize, self).__init__(always_apply, p)
        self.mask = mask
        self.mask_params = mask_params
        self.range = to_tuple(range, 0)

    def apply(
        self, image: np.ndarray, mask: Union[None, np.ndarray] = None, **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.equalize(image, mask=mask, hist_range=self.range)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self) -> List[str]:
        """Returns a list of target names (e.g. 'image') that are needed as a parameter input
        to other `apply` methods (e.g. apply_to_bboxes(..., image = image))
        """
        return ["image"] + list(self.mask_params)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("range",)

class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        max_brightness (int,float,None): If not None, adjust contrast by specified maximum and clip to maximum,
                else adjust contrast by image mean. Default: None
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        max_brightness: Union[int, float, None] = None,
        brightness_limit: Union[float, Tuple[float, float]] = 0.2,
        contrast_limit: Union[float, Tuple[float, float]] = 0.2,
        always_apply: bool = False,
        p: bool = 0.5,
    ):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.max_brightness = max_brightness

    def apply(
        self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0, **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.brightness_contrast_adjust(img, alpha, beta, self.max_brightness)

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "alpha": 1.0
            + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0
            + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("brightness_limit", "contrast_limit", "max_brightness")


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        apply_to_channel_idx (int, None): If not None, then only only noise is applied on the specified channel index. Default: None
        per_channel (bool): if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Ignored if apply_to_channel_idx is not None. Default: True
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, int16, float32
    """

    def __init__(
        self,
        var_limit: Union[float, Tuple[float, float]] = (10.0, 50.0),
        mean: float = 0,
        apply_to_channel_idx: Union[None, int] = None,
        per_channel: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GaussNoise, self).__init__(always_apply, p)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(
                    type(var_limit)
                )
            )

        if apply_to_channel_idx is not None:
            if isinstance(apply_to_channel_idx, int):
                raise TypeError(
                    "Expected apply_to_channel_idx to be one of (None, int), got {}".format(
                        type(apply_to_channel_idx)
                    )
                )
            if apply_to_channel_idx < 0:
                raise ValueError("apply_to_channel_idx should be non negative")

        self.mean = mean
        self.per_channel = per_channel
        self.apply_to_channel_idx = apply_to_channel_idx

    def apply(
        self, img: np.ndarray, gauss: Union[None, np.ndarray] = None, **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5

        if self.apply_to_channel_idx is not None:
            if image.ndim != 4:
                raise RuntimeError(
                    "Expected image to be of shape (H,W,D,C) with argument 'apply_to_channel_idx' not None, got shape {}".format(
                        image.shape
                    )
                )
            if self.apply_to_channel_idx >= image.shape[-1]:
                raise IndexError(
                    "Index {} out of range for image of shape {}".format(
                        self.apply_to_channel_idx, image.shape
                    )
                )

            gauss = np.zeros_like(image, dtype=np.float32)
            gauss[..., self.apply_to_channel_idx] = random_utils.normal(
                self.mean, sigma, image.shape[:3]
            )

        elif self.per_channel:
            gauss = random_utils.normal(self.mean, sigma, image.shape)
        else:
            gauss = random_utils.normal(self.mean, sigma, image.shape[:3])
            if len(image.shape) == 4:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss}

    @property
    def targets_as_params(self) -> List[str]:
        """Returns a list of target names (e.g. 'image') that are needed as a parameter input
        to other `apply` methods (e.g. apply_to_bboxes(..., image = image))
        """
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("var_limit", "apply_to_channel_idx", "per_channel", "mean")


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from the maximum value for the input image dtype.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, int16, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.invert(img)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()


class RandomGamma(ImageOnlyTransform):
    """
    Args:
        gamma_limit (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, gamma_limit=(80, 120), always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)

    def apply(self, img: np.ndarray, gamma: float = 1, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("gamma_limit",)


class ToFloat(ImageOnlyTransform):
    """Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
    If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
    image.

    See Also:
        :class:`~albumentations.augmentations.transforms.FromFloat`

    Args:
        min_value (float): minimum possible input value. Default: None.
        max_value (float): maximum possible input value. Default: None.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        any type

    """

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        always_apply=False,
        p=1.0,
    ):
        super(ToFloat, self).__init__(always_apply, p)
        self.max_value = max_value
        self.min_value = min_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.to_float(img, self.min_value, self.max_value)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("max_value", "min_value")


class FromFloat(ImageOnlyTransform):
    """Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument.

    This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.

    Args:
        min_value (float): minimum possible input value. Default: None.
        max_value (float): maximum possible input value. Default: None.
        dtype (string or numpy data type): data type of the output. See the `'Data types' page from the NumPy docs`_.
            Default: 'int16'.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    .. _'Data types' page from the NumPy docs:
       https://docs.scipy.org/doc/numpy/user/basics.types.html
    """

    def __init__(
        self,
        dtype: str = "int16",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        always_apply=False,
        p=1.0,
    ):
        super(FromFloat, self).__init__(always_apply, p)
        self.dtype = np.dtype(dtype)
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.from_float(img, self.dtype, self.min_value, self.max_value)

    def get_transform_init_args(self) -> Dict[str, Any]:
        """Returns initialization arguments (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1' : 1, 'arg2': 2))"""
        return {
            "dtype": self.dtype.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.

    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  upper bound on the image scale. Should be < 1.
        interpolation (int, dict, Interpolation): scipy interpolation method (e.g. `dicaugment.INTER_NEAREST`). Could be:

            - Single Scipy interpolation flag: The selected method will be used for both downscale and upscale.

            - `dict` of flags: Dictionary with keys 'downscale' and 'upscale' specifying the interpolation flags for each operation.

            - `Interpolation` object: Downscale.Interpolation object with flags for both downscale and upscale.

            Default: `Interpolation(downscale=dicaugment.INTER_NEAREST, upscale=dicaugment.INTER_NEAREST)`
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, int16, int32, float32
    """

    class Interpolation:
        def __init__(
            self, *, downscale: int = INTER_NEAREST, upscale: int = INTER_NEAREST
        ):
            self.downscale = downscale
            self.upscale = upscale

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.25,
        interpolation: Optional[Union[int, Interpolation, Dict[str, int]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(Downscale, self).__init__(always_apply, p)
        if interpolation is None:
            self.interpolation = self.Interpolation()
            warnings.warn(
                "Using default interpolation INTER_NEAREST, which is sub-optimal."
                "Please specify interpolation mode for downscale and upscale explicitly."
            )
        elif isinstance(interpolation, int):
            self.interpolation = self.Interpolation(
                downscale=interpolation, upscale=interpolation
            )
        elif isinstance(interpolation, self.Interpolation):
            self.interpolation = interpolation
        elif isinstance(interpolation, dict):
            self.interpolation = self.Interpolation(**interpolation)
        else:
            raise ValueError(
                "Wrong interpolation data type. Supported types: `Optional[Union[int, Interpolation, Dict[str, int]]]`."
                f" Got: {type(interpolation)}"
            )

        if scale_min > scale_max:
            raise ValueError(
                "Expected scale_min be less or equal scale_max, got {} {}".format(
                    scale_min, scale_max
                )
            )
        if scale_max >= 1:
            raise ValueError(
                "Expected scale_max to be less than 1, got {}".format(scale_max)
            )
        self.scale_min = scale_min
        self.scale_max = scale_max

    def apply(
        self, img: np.ndarray, scale: Optional[float] = None, **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation.downscale,
            up_interpolation=self.interpolation.upscale,
        )

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {"scale": random.uniform(self.scale_min, self.scale_max)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return "scale_min", "scale_max"

    def _to_dict(self) -> Dict[str, Any]:
        result = super()._to_dict()
        result["interpolation"] = {
            "upscale": self.interpolation.upscale,
            "downscale": self.interpolation.downscale,
        }
        return result


class Sharpen(ImageOnlyTransform):
    """
    Sharpen the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness ((float, float)): range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        cval (int,float): The fill value when mode = `constant`. Default: 0
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self,
        alpha: Union[Tuple[float, float], float] = (0.2, 0.5),
        lightness: Union[Tuple[float, float], float] = (0.5, 1.0),
        mode: str = "constant",
        cval: Union[float, int] = 0,
        always_apply=False,
        p=0.5,
    ):
        super(Sharpen, self).__init__(always_apply, p)
        self.alpha = self.__check_values(
            to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0)
        )
        self.lightness = self.__check_values(to_tuple(lightness, 0.0), name="lightness")
        self.mode = mode
        self.cval = cval

        if self.mode not in {"reflect", "constant", "nearest", "mirror", "wrap"}:
            raise ValueError(
                "Expected mode to be one of ('reflect', 'constant', 'nearest', 'mirror', 'wrap'), got {}".format(
                    self.mode
                )
            )

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_sharpening_matrix(alpha_sample, lightness_sample, ksize=3):
        if ksize % 2 != 1:
            raise ValueError("expected ksize to be an odd number, got {}".format(ksize))

        # matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_nochange = np.zeros((ksize, ksize, ksize), dtype=np.float32)
        matrix_nochange[ksize // 2, ksize // 2, ksize // 2] = 1.0

        matrix_effect = np.ones((ksize, ksize, ksize), dtype=np.float32)
        matrix_effect[ksize // 2, ksize // 2, ksize // 2] = -(
            np.sum(matrix_effect) - 1 + lightness_sample
        )
        matrix_effect *= -1.0

        # matrix_effect = np.array(
        #     [[-1, -1, -1], [-1, 8 + lightness_sample, -1], [-1, -1, -1]],
        #     dtype=np.float32,
        # )

        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(
            alpha_sample=alpha, lightness_sample=lightness
        )
        return {"sharpening_matrix": sharpening_matrix}

    def apply(
        self,
        img: np.ndarray,
        sharpening_matrix: Union[None, np.ndarray] = None,
        **params,
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.convolve(img, sharpening_matrix, self.mode, self.cval)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("alpha", "lightness", "mode", "cval")


class UnsharpMask(ImageOnlyTransform):
    """
    Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * 4 * 2) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha (float, (float, float)): range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold (float): Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 1]. Default: 0.05.
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

    Reference:
        https://arxiv.org/pdf/2107.10833.pdf

    Targets:
        image
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (3, 7),
        sigma_limit: Union[float, Sequence[float]] = 0.0,
        alpha: Union[float, Sequence[float]] = (0.2, 0.5),
        threshold: float = 0.05,
        mode: str = "constant",
        cval: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(UnsharpMask, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.sigma_limit = self.__check_values(
            to_tuple(sigma_limit, 0.0), name="sigma_limit"
        )
        self.alpha = self.__check_values(
            to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0)
        )
        self.threshold = threshold
        self.mode = mode
        self.cval = cval

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            raise ValueError(
                "blur_limit and sigma_limit minimum value can not be both equal to 0."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("UnsharpMask supports only odd blur limits.")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self) -> Dict[str, Any]:
        """Returns parameters needed for the `apply` methods"""
        return {
            "ksize": random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2),
            "sigma": random.uniform(*self.sigma_limit),
            "alpha": random.uniform(*self.alpha),
        }

    def apply(
        self,
        img: np.ndarray,
        ksize: int = 3,
        sigma: float = 0,
        alpha: float = 0.2,
        **params,
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.unsharp_mask(
            img,
            ksize,
            sigma=sigma,
            alpha=alpha,
            threshold=self.threshold,
            mode=self.mode,
            cval=self.cval,
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("blur_limit", "sigma_limit", "alpha", "threshold")


class PixelDropout(DualTransform):
    """
    Set pixels to 0 with some probability.

    Args:
        dropout_prob (float): pixel drop probability. Default: 0.01
        per_channel (bool): if set to `True` drop mask will be sampled fo each channel,
            otherwise the same mask will be sampled for all channels. Default: False
        drop_value (number or sequence of numbers or None): Value that will be set in dropped place. If set to None value will be sampled randomly, default ranges will be used:
            - uint8: [0, 255]
            - uint16: [0, 65535]
            - uint32: [0, 4294967295]
            - int16 - [-32768, 32767]
            - int32 - [-2147483648, 2147483647]
            - float, double - [0, 1]
            Default: 0
        mask_drop_value (number or sequence of numbers or None): Value that will be set in dropped place in masks.
            If set to None masks will be unchanged. Default: 0
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask

    Image types:
        any
    """

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: Optional[Union[float, Sequence[float]]] = 0,
        mask_drop_value: Optional[Union[float, Sequence[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

        if self.mask_drop_value is not None and self.per_channel:
            raise ValueError("PixelDropout supports mask only with per_channel=False")

    def apply(
        self,
        img: np.ndarray,
        drop_mask: np.ndarray = np.array(None),
        drop_value: Union[float, Sequence[float]] = (),
        **params,
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return F.pixel_dropout(img, drop_mask, drop_value)

    def apply_to_mask(
        self, img: np.ndarray, drop_mask: np.ndarray = np.array(None), **params
    ) -> np.ndarray:
        if self.mask_drop_value is None:
            return img

        if img.ndim == 3:
            drop_mask = np.squeeze(drop_mask)

        return F.pixel_dropout(img, drop_mask, self.mask_drop_value)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Applies the augmentation to a bbox"""
        return bbox

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Applies the augmentation to a keypoint"""
        return keypoint

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        img = params["image"]
        shape = img.shape if self.per_channel and img.ndim == 4 else img.shape[:3]

        rnd = np.random.RandomState(random.randint(0, 1 << 31))
        # Use choice to create boolean matrix, if we will use binomial after that we will need type conversion
        drop_mask = rnd.choice(
            [True, False], shape, p=[self.dropout_prob, 1 - self.dropout_prob]
        )

        if drop_mask.ndim != img.ndim:
            drop_mask = np.expand_dims(drop_mask, -1)

        drop_value: Union[float, Sequence[float], np.ndarray]

        if self.drop_value is None:
            drop_shape = 1 if is_grayscale_image(img) else int(img.shape[-1])

            if img.dtype in (np.uint8, np.uint16, np.uint32, np.int16, np.int32):
                drop_value = rnd.randint(
                    F.MIN_VALUES_BY_DTYPE[img.dtype],
                    int(F.MAX_VALUES_BY_DTYPE[img.dtype]),
                    drop_shape,
                    img.dtype,
                )
            elif img.dtype in [np.float32, np.double]:
                drop_value = rnd.uniform(0, 1, drop_shape).astype(img.dtype)
            else:
                raise ValueError(f"Unsupported dtype: {img.dtype}")
        else:
            drop_value = self.drop_value

        return {"drop_mask": drop_mask, "drop_value": drop_value}

    @property
    def targets_as_params(self) -> List[str]:
        """Returns a list of target names (e.g. 'image') that are needed as a parameter input
        to other `apply` methods (e.g. apply_to_bboxes(..., image = image))
        """
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ("dropout_prob", "per_channel", "drop_value", "mask_drop_value")
