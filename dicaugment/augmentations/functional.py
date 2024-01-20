from __future__ import division

from typing import Optional, Sequence, Union, Tuple, Any
from warnings import warn

import cv2
import numpy as np
from scipy import ndimage

from dicaugment import random_utils
from dicaugment.augmentations.utils import (
    MAX_VALUES_BY_DTYPE,
    MIN_VALUES_BY_DTYPE,
    _maybe_process_in_chunks,
    _maybe_process_by_channel,
    clip,
    clipped,
    ensure_contiguous,
    is_grayscale_image,
    is_rgb_image,
    is_multispectral_image,
    is_uint8_or_float32,
    non_rgb_warning,
    preserve_channel_dim,
    preserve_shape,
)

from ..core.transforms_interface import (
    INTER_NEAREST,
    INTER_LINEAR,
    INTER_QUADRATIC,
    INTER_CUBIC,
    INTER_QUARTIC,
    INTER_QUINTIC,
)

__all__ = [
    "brightness_contrast_adjust",
    "convolve",
    "downscale",
    "equalize",
    "from_float",
    "gamma_transform",
    "gauss_noise",
    "invert",
    "multiply",
    "noop",
    "normalize",
    "to_float",
    "unsharp_mask",
]


def normalize(
        img: np.ndarray,
        mean: Union[float, np.ndarray, None],
        std: Union[float, np.ndarray, None]
    ) -> np.ndarray:
    """
    Normalizes an image by the formula `img = (img - mean) / (std)`.
    
    Args:
        img (np.ndarray): an image
        mean (float, np.ndarray, None): The offset for the image. 
            If None, mean is calculated as the mean of the image. If np.ndarray, operation can be broadcast across dimensions.
        std (float, np.ndarray, None): The standard deviation to divide the image by. 
            If None, std is calculated as the std of the image. If np.ndarray, operation can be broadcast across dimensions.
    
    """
    ndim = img.ndim
    axis = None if ndim == 3 else tuple(range(ndim - 1))

    # if max_pixel_value == None:
    #     max_pixel_value = np.max(img, axis = axis)

    if mean is None:
        mean = np.mean(img, axis=axis)

    if std is None:
        std = np.std(img, axis=axis)

    mean = np.array(mean, dtype=np.float32)
    # mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    # std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


@preserve_shape
def posterize(img, bits):
    """Reduce the number of bits for each color channel.

    Args:
        img (numpy.ndarray): image to posterize.
        bits (int): number of high bits. Must be in range [0, 8]

    Returns:
        numpy.ndarray: Image with reduced color channels.

    """
    bits = np.uint8(bits)

    dtypes = {
        "uint8": (np.uint8, 8),
        "uint16": (np.uint16, 8),
        "int16": (np.int16, 16),
        "int32": (np.int32, 32),
    }

    if img.dtype.name not in dtypes.keys():
        raise TypeError(
            "dtype must be one of {}, got {}".format(
                tuple(dtypes.keys()), img.dtype.name
            )
        )

    dtype_func, max_bits = dtypes[img.dtype.name]

    if np.any((bits < 0) | (bits > max_bits)):
        raise ValueError(
            "bits must be in range [0, {}] for {} data type".format(
                max_bits, img.dtype.name
            )
        )

    if not bits.shape or len(bits) == 1:
        if bits == 0:
            return np.zeros_like(img)
        if bits == max_bits:
            return img.copy()

    if img.dtype.name == "uint8":
        if not bits.shape or len(bits) == 1:
            lut = np.arange(0, 256, dtype=np.uint8)
            mask = ~np.uint8(2 ** (8 - bits) - 1)
            lut &= mask

            return cv2.LUT(img, lut)

        if not is_rgb_image(img) and not is_multispectral_image(img):
            raise TypeError(
                "If bits is iterable, then image must be RGB or Multispectral"
            )

        result_img = np.empty_like(img)
        for i, channel_bits in enumerate(bits):
            if channel_bits == 0:
                result_img[..., i] = np.zeros_like(img[..., i])
            elif channel_bits == 8:
                result_img[..., i] = img[..., i].copy()
            else:
                lut = np.arange(0, 256, dtype=np.uint8)
                mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
                lut &= mask

                result_img[..., i] = cv2.LUT(img[..., i], lut)

        return result_img

    if not bits.shape or len(bits) == 1:
        mask = ~dtype_func(2 ** (max_bits - bits) - 1)
        return img.copy() & mask

    if not is_rgb_image(img) and not is_multispectral_image(img):
        raise TypeError("If bits is iterable, then image must be RGB or Multispectral")

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits):
        if channel_bits == 0:
            result_img[..., i] = np.zeros_like(img[..., i])
        elif channel_bits == max_bits:
            result_img[..., i] = img[..., i].copy()
        else:
            mask = ~dtype_func(2 ** (max_bits - channel_bits) - 1)
            result_img[..., i] = img[..., i].copy() & mask

    return result_img

def _calcHist(
    img: np.ndarray, mask: Union[np.ndarray, None], nbins: int, hist_range: Tuple
):
    if not mask:
        mask = np.ones_like(img, dtype=np.bool_)

    bins = np.linspace(hist_range[0], hist_range[1] + 1, nbins + 1)

    return np.histogram(img[mask.astype(np.bool_)], bins=bins)[0]


def _equalize_cv(img, hist_range, mask=None):
    lo, hi = hist_range
    histogram = sum(map(lambda x: _calcHist(x, mask, hi - lo, hist_range), img)).ravel()

    total = np.sum(histogram)
    histogram = histogram / total
    cumsum = (np.cumsum(histogram) * (hi - lo)) + lo

    lut = {}

    for i in range(lo, hi):
        lut[i] = clip(round(cumsum[i - lo]), img.dtype, lo, hi)

    return np.vectorize(lambda x: lut.get(x, x))(img)


@preserve_channel_dim
def equalize(img, hist_range=None, mask=None):
    """Equalize the image histogram.

    Args:
        img (numpy.ndarray): image.
        hist_range (tuple): The histogram range
        mask (numpy.ndarray): An optional mask.  If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array.

    Returns:
        numpy.ndarray: Equalized image.

    """
    if img.dtype not in {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("int32"),
    }:
        raise TypeError("Image must have int or uint type")

    if mask is not None:
        if not is_grayscale_image(mask) and is_grayscale_image(img):
            raise ValueError(
                "Wrong mask shape. Image shape: {}. "
                "Mask shape: {}".format(img.shape, mask.shape)
            )
        # if not by_channels and not is_grayscale_image(mask):
        #     raise ValueError(
        #         "When by_channels=False only 1-channel mask ared supported. " "Mask shape: {}".format(mask.shape)
        #     )

    if hist_range is None:
        hist_range = (0, np.max(img))

    if mask is not None:
        mask = mask.astype(np.bool_)

    if is_grayscale_image(img):
        return _equalize_cv(img, hist_range, mask)

    # if not by_channels:
    #     result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    #     result_img[..., 0] = function(result_img[..., 0], mask)
    #     return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(img.shape[-1]):
        if mask is None:
            _mask = None
        elif is_grayscale_image(mask):
            _mask = mask
        else:
            _mask = mask[..., i]

        result_img[..., i] = _equalize_cv(img[..., i], hist_range, _mask)

    return result_img


@preserve_shape
def move_tone_curve(img, low_y, high_y):
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        img (numpy.ndarray): RGB or grayscale image.
        low_y (float): y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float): y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
    """
    input_dtype = img.dtype

    if low_y < 0 or low_y > 1:
        raise ValueError("low_shift must be in range [0, 1]")
    if high_y < 0 or high_y > 1:
        raise ValueError("high_shift must be in range [0, 1]")

    if input_dtype != np.uint8:
        raise ValueError("Unsupported image type {}".format(input_dtype))

    t = np.linspace(0.0, 1.0, 256)

    # Defines responze of a four-point bezier curve
    def evaluate_bez(t):
        return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3

    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

    lut_fn = _maybe_process_in_chunks(cv2.LUT, lut=remapping)
    img = lut_fn(img)
    return img


@clipped
def _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        return img + r_shift

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = img[..., i] + shift

    return result_img


def _shift_image_uint8(img, value):
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    lut = np.arange(0, max_value + 1).astype("float32")
    lut += value

    lut = np.clip(lut, 0, max_value).astype(img.dtype)
    return cv2.LUT(img, lut)


@preserve_shape
def _shift_rgb_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        h, w, c = img.shape
        img = img.reshape([h, w * c])

        return _shift_image_uint8(img, r_shift)

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = _shift_image_uint8(img[..., i], shift)

    return result_img


def shift_rgb(img, r_shift, g_shift, b_shift):
    if img.dtype == np.uint8:
        return _shift_rgb_uint8(img, r_shift, g_shift, b_shift)

    return _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift)


@clipped
def linear_transformation_rgb(img, transformation_matrix):
    result_img = cv2.transform(img, transformation_matrix)

    return result_img


@preserve_channel_dim
def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


@preserve_shape
@clipped
def convolve(
    img: np.ndarray,
    kernel: np.ndarray,
    mode: str = "constant",
    cval: Union[int,float] = 0
    ) -> np.ndarray:
    """Applies a convolutional kernel to an image
    
    Args:
        img (np.ndarray): an image
        kernel (np.ndarray): a kernel to convolve over image
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

                    * `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
                    * `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
                    * `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
                    * `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
                    * `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

                    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

                    Default: `constant`
        cval (int,float): The fill value when mode = `constant`. Default: 0

    Returns:
        np.ndarray: the convolved image
    """
    convolve_fn = _maybe_process_by_channel(
        ndimage.convolve, weights=kernel, mode=mode, cval=cval
    )
    return convolve_fn(img)


@preserve_shape
def image_compression(img, quality, image_type):
    if image_type in [".jpeg", ".jpg"]:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        NotImplementedError(
            "Only '.jpg' and '.webp' compression transforms are implemented. "
        )

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warn(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            "{} is used as input.".format(input_dtype),
            UserWarning,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for image augmentation".format(input_dtype)
        )

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    return img


@preserve_shape
def add_snow(img, snow_point, brightness_coeff):
    """Bleaches out pixels, imitation snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        snow_point: Number of show points.
        brightness_coeff: Brightness coefficient.

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    snow_point *= 127.5  # = 255 / 2
    snow_point += 85  # = 255 / 3

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for RandomSnow augmentation".format(input_dtype)
        )

    image_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float32)

    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] *= brightness_coeff

    image_HLS[:, :, 1] = clip(image_HLS[:, :, 1], np.uint8, 255)

    image_HLS = np.array(image_HLS, dtype=np.uint8)

    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_RGB = to_float(image_RGB, max_value=255)

    return image_RGB


@preserve_shape
def add_rain(
    img,
    slant,
    drop_length,
    drop_width,
    drop_color,
    blur_value,
    brightness_coefficient,
    rain_drops,
):
    """

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        slant (int):
        drop_length:
        drop_width:
        drop_color:
        blur_value (int): Rainy view are blurry.
        brightness_coefficient (float): Rainy days are usually shady.
        rain_drops:

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for RandomRain augmentation".format(input_dtype)
        )

    image = img.copy()

    for rain_drop_x0, rain_drop_y0 in rain_drops:
        rain_drop_x1 = rain_drop_x0 + slant
        rain_drop_y1 = rain_drop_y0 + drop_length

        cv2.line(
            image,
            (rain_drop_x0, rain_drop_y0),
            (rain_drop_x1, rain_drop_y1),
            drop_color,
            drop_width,
        )

    image = cv2.blur(image, (blur_value, blur_value))  # rainy view are blurry
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 2] *= brightness_coefficient

    image_rgb = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_shape
def add_fog(img, fog_coef, alpha_coef, haze_list):
    """Add fog to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        fog_coef (float): Fog coefficient.
        alpha_coef (float): Alpha coefficient.
        haze_list (list):

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for RandomFog augmentation".format(input_dtype)
        )

    width = img.shape[1]

    hw = max(int(width // 3 * fog_coef), 10)

    for haze_points in haze_list:
        x, y = haze_points
        overlay = img.copy()
        output = img.copy()
        alpha = alpha_coef * fog_coef
        rad = hw // 2
        point = (x + hw // 2, y + hw // 2)
        cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        img = output.copy()

    image_rgb = cv2.blur(img, (hw // 10, hw // 10))

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_shape
def add_sun_flare(img, flare_center_x, flare_center_y, src_radius, src_color, circles):
    """Add sun flare.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        flare_center_x (float):
        flare_center_y (float):
        src_radius:
        src_color (int, int, int):
        circles (list):

    Returns:
        numpy.ndarray:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for RandomSunFlareaugmentation".format(input_dtype)
        )

    overlay = img.copy()
    output = img.copy()

    for alpha, (x, y), rad3, (r_color, g_color, b_color) in circles:
        cv2.circle(overlay, (x, y), rad3, (r_color, g_color, b_color), -1)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    point = (int(flare_center_x), int(flare_center_y))

    overlay = output.copy()
    num_times = src_radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, src_radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = (
            alpha[num_times - i - 1]
            * alpha[num_times - i - 1]
            * alpha[num_times - i - 1]
        )
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)

    image_rgb = output

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@ensure_contiguous
@preserve_shape
def add_shadow(img, vertices_list):
    """Add shadows to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        vertices_list (list):

    Returns:
        numpy.ndarray:

    """
    non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for RandomShadow augmentation".format(input_dtype)
        )

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(img)

    # adding all shadow polygons on empty mask, single 255 denotes only red channel
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)

    # if red channel is hot, image's "Lightness" channel's brightness is lowered
    red_max_value_ind = mask[:, :, 0] == 255
    image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.5

    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@ensure_contiguous
@preserve_shape
def add_gravel(img: np.ndarray, gravels: list):
    """Add gravel to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): image to add gravel to
        gravels (list): list of gravel parameters. (float, float, float, float):
            (top-left x, top-left y, bottom-right x, bottom right y)

    Returns:
        numpy.ndarray:
    """
    non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(
            "Unexpected dtype {} for AddGravel augmentation".format(input_dtype)
        )

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for gravel in gravels:
        y1, y2, x1, x2, sat = gravel
        image_hls[x1:x2, y1:y2, 1] = sat

    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


def invert(img: np.ndarray) -> np.ndarray:
    """
    Inverts the pixel values of an image.

    Args:
        img (np.ndarray): an image
    """
    if img.dtype == np.float32 and np.max(img) > 1.0:
        warn(
            "Images with dtype float32 are expected to remain in the range of [0,1]. Returned image will contain negative values",
            UserWarning,
        )

    return MAX_VALUES_BY_DTYPE[img.dtype] - (img + MIN_VALUES_BY_DTYPE[img.dtype])


def channel_shuffle(img, channels_shuffled):
    img = img[..., channels_shuffled]
    return img


@preserve_shape
@clipped
def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Performs a Gamma correction on an image.

    Args:
        img (np.ndarray): an image
        gamma (float): gamma parameter
    """
    return np.power(img, gamma)


@clipped
def gauss_noise(image, gauss):
    """
    Adds noise to an image.

    Args:
        img (np.ndarray): an image
        guass (np.ndarray): guassian noise parameter
    """
    image = image.astype("float32")
    return image + gauss


@clipped
def _brightness_contrast_adjust(img, alpha=1, beta=0, max_brightness=None):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if max_brightness is not None:
            img += beta * max_brightness
        else:
            img += beta * np.mean(img)

    if max_brightness is not None:
        img = np.clip(img, MIN_VALUES_BY_DTYPE[dtype], max_brightness)

    return img


def brightness_contrast_adjust(
        img: np.ndarray,
        alpha: Union[float,int] = 1, 
        beta: Union[float,int] = 0, 
        max_brightness: Optional[Union[float,int]] = None
    ) -> np.ndarray:
    """
    Adjusts the brightness and/or contrast of an image

    Args:
        img (np.ndarray): an image
        alpha (int,float): The contrast parameter
        beta (int,float): The brightness parameter
        max_brightness (int,float,None): If not None, adjust contrast by specified maximum and clip to maximum,
                else adjust contrast by image mean. Default: None
    """
    return _brightness_contrast_adjust(img, alpha, beta, max_brightness)


@clipped
def iso_noise(image, color_shift=0.05, intensity=0.5, random_state=None, **kwargs):
    """
    Apply poisson noise to image to simulate camera sensor noise.

    Args:
        image (numpy.ndarray): Input image, currently, only RGB, uint8 images are supported.
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
        random_state:
        **kwargs:

    Returns:
        numpy.ndarray: Noised image

    """
    if image.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")
    if not is_rgb_image(image):
        raise TypeError("Image must be RGB")

    one_over_255 = float(1.0 / 255.0)
    image = np.multiply(image, one_over_255, dtype=np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_utils.poisson(
        stddev[1] * intensity * 255, size=hls.shape[:2], random_state=random_state
    )
    color_noise = random_utils.normal(
        0, color_shift * 360 * intensity, size=hls.shape[:2], random_state=random_state
    )

    hue = hls[..., 0]
    hue += color_noise
    hue[hue < 0] += 360
    hue[hue > 360] -= 360

    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
    return image.astype(np.uint8)


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def gray_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


@preserve_shape
def downscale(
    img: np.ndarray,
    scale: float,
    down_interpolation: int = INTER_LINEAR,
    up_interpolation: int = INTER_LINEAR
) -> np.ndarray:
    """
    Decreases image quality by downscaling and upscaling back.

    Args:
        img (np.ndarray): an image
        scale (float): the scale to downsize to
        down_interpolation (int, Interpolation): scipy interpolation method (e.g. `dicaugment.INTER_NEAREST`)
        up_interpolation (int, Interpolation): scipy interpolation method (e.g. `dicaugment.INTER_NEAREST`)
    """
    h, w, d = img.shape[:3]

    if img.ndim == 4:
        upscaled = np.zeros_like(img)
        for i in range(img.shape[-1]):
            downscaled = ndimage.zoom(img[..., i], scale, order=down_interpolation)
            inv_scale_h = h / downscaled.shape[0]
            inv_scale_w = w / downscaled.shape[1]
            inv_scale_d = d / downscaled.shape[2]
            inv_scale = (inv_scale_h, inv_scale_w, inv_scale_d)
            upscaled[..., i] = ndimage.zoom(
                downscaled, inv_scale, order=up_interpolation
            )

    else:
        downscaled = ndimage.zoom(img, scale, order=down_interpolation)
        inv_scale_h = h / downscaled.shape[0]
        inv_scale_w = w / downscaled.shape[1]
        inv_scale_d = d / downscaled.shape[2]
        inv_scale = (inv_scale_h, inv_scale_w, inv_scale_d)
        upscaled = ndimage.zoom(downscaled, inv_scale, order=up_interpolation)

    return upscaled


def to_float(img, min_value=None, max_value=None):
    """
    Convert an image to a floating point image based on current dtype
    
    Args:
        img (np.ndarray): an image
        min_value (int,float,None): Optional custom minimum value of dtype. Maps this value to the lower bound of `float32` (0.0).
        max_value (int,float,None): Optional custom maximum value of dtype. Maps this value to the upper bound of `float32` (1.0).

    Returns:
        np.ndarray: image cast to `float32`

    Raises:
        RuntimeError: if image dtype is not one of {`uint8`, `uint16`, `uint32`, `float32`, `int16`, `int32`, `float64`}

    """
    if max_value is None or min_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
            min_value = MIN_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the minimum and maximum value for dtype {}. You need to specify the minimum and maximum value manually by "
                "passing the min_value and max_value arguments".format(img.dtype)
            )
    return (img.astype("float32") - min_value) / (max_value - min_value)


def from_float(
        img: np.ndarray,
        dtype: str,
        min_value: Optional[Union[int,float]] = None,
        max_value: Optional[Union[int,float]] = None
    ) -> np.ndarray:
    """
    Convert an image from a floating point image, to the specified dtype
    
    Args:
        img (np.ndarray): an image
        dtype (str): a dtype to cast to. Must be one of {`uint8`, `uint16`, `uint32`, `float32`, `int16`, `int32`, `float64`}
        min_value (int,float,None): Optional custom minimum value of dtype. Maps lower bound of `float32` (0.0) to this value.
        max_value (int,float,None): Optional custom maximum value of dtype. Maps upper bound of `float32` (1.0) to this value.

    Returns:
        np.ndarray: image cast to `dtype`

    Raises:
        RuntimeError: if dtype is not one of {`uint8`, `uint16`, `uint32`, `float32`, `int16`, `int32`, `float64`}

    """
    if max_value is None or min_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[np.dtype(dtype)]
            min_value = MIN_VALUES_BY_DTYPE[np.dtype(dtype)]
        except KeyError:
            raise RuntimeError(
                "Can't infer the minimum and maximum value for dtype {}. You need to specify the minimum and maximum value manually by "
                "passing the min_value and max_value arguments".format(dtype)
            )
    return (img * (max_value - min_value) + min_value).astype(dtype)


def noop(input_obj: Any, **params):  # skipcq: PYL-W0613
    """Does nothing. Returns the input object"""
    return input_obj


def swap_tiles_on_image(image, tiles):
    """
    Swap tiles on image.

    Args:
        image (np.ndarray): Input image.
        tiles (np.ndarray): array of tuples(
            current_left_up_corner_row, current_left_up_corner_col,
            old_left_up_corner_row, old_left_up_corner_col,
            height_tile, width_tile)

    Returns:
        np.ndarray: Output image.

    """
    new_image = image.copy()

    for tile in tiles:
        new_image[tile[0] : tile[0] + tile[4], tile[1] : tile[1] + tile[5]] = image[
            tile[2] : tile[2] + tile[4], tile[3] : tile[3] + tile[5]
        ]

    return new_image


@clipped
def _multiply_uint8(img, multiplier):
    img = img.astype(np.float32)
    return np.multiply(img, multiplier)


@preserve_shape
def _multiply_uint8_optimized(img, multiplier):
    if is_grayscale_image(img):
        multiplier = multiplier[0]
        lut = np.arange(0, 256, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img if img.ndim == 3 else img[:3])

    channels = img.shape[-1]
    lut = [np.arange(0, 256, dtype=np.float32)] * channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])

    images = []
    for i in range(channels):
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut[:, i])
        images.append(func(img[..., i]))
    return np.stack(images, axis=-1)


@clipped
def _multiply_non_uint8(img, multiplier):
    return img * multiplier


def multiply(img, multiplier):
    """
    Args:
        img (numpy.ndarray): Image.
        multiplier (numpy.ndarray): Multiplier coefficient.

    Returns:
        numpy.ndarray: Image multiplied by `multiplier` coefficient.

    """
    if img.dtype == np.uint8:
        if len(multiplier.shape) == 1:
            return _multiply_uint8_optimized(img, multiplier)

        return _multiply_uint8(img, multiplier)

    return _multiply_non_uint8(img, multiplier)


def bbox_from_mask(mask):
    """Create bounding box from binary mask (fast version)

    Args:
        mask (numpy.ndarray): binary mask.

    Returns:
        tuple: A bounding box tuple `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """
    rows = np.any(mask, axis=1)
    if not rows.any():
        return -1, -1, -1, -1, -1, -1
    cols = np.any(mask, axis=0)
    slices = np.any(mask, axis=2)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    z_min, z_max = np.where(slices)[0][[0, -1]]
    return x_min, y_min, z_min, x_max + 1, y_max + 1, z_max + 1


def mask_from_bbox(img, bbox):
    """Create binary mask from bounding box

    Args:
        img (numpy.ndarray): input image
        bbox: A bounding box tuple `(x_min, y_min, z_min, x_max, y_max, z_max)`

    Returns:
        mask (numpy.ndarray): binary mask

    """

    mask = np.zeros(img.shape[:3], dtype=np.uint8)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    mask[y_min:y_max, x_min:x_max, z_min:z_max] = 1
    return mask


def fancy_pca(img, alpha=0.1):
    """Perform 'Fancy PCA' augmentation from:
    http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    Args:
        img (numpy.ndarray): numpy array with (h, w, rgb) shape, as ints between 0-255
        alpha (float): how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1

    Returns:
        numpy.ndarray: numpy image-like array as uint8 range(0, 255)

    """
    if not is_rgb_image(img) or img.dtype != np.uint8:
        raise TypeError("Image must be RGB image in uint8 format.")

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    return orig_img


def _adjust_brightness_torchvision_uint8(img, factor):
    lut = np.arange(0, 256) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


@preserve_shape
def adjust_brightness_torchvision(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        return _adjust_brightness_torchvision_uint8(img, factor)

    return clip(img * factor, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_contrast_torchvision_uint8(img, factor, mean):
    lut = np.arange(0, 256) * factor
    lut = lut + mean * (1 - factor)
    lut = clip(lut, img.dtype, 255)

    return cv2.LUT(img, lut)


@preserve_shape
def adjust_contrast_torchvision(img, factor):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    if img.dtype == np.uint8:
        return _adjust_contrast_torchvision_uint8(img, factor, mean)

    return clip(
        img.astype(np.float32) * factor + mean * (1 - factor),
        img.dtype,
        MAX_VALUES_BY_DTYPE[img.dtype],
    )


@preserve_shape
def adjust_saturation_torchvision(img, factor, gamma=0):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_hue_torchvision_uint8(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img, factor):
    if is_grayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


@clipped
def add_weighted(img1, alpha, img2, beta):
    return img1.astype(float) * alpha + img2.astype(float) * beta


@clipped
@preserve_shape
def unsharp_mask(
    image: np.ndarray,
    ksize: int,
    sigma: float = 0.0,
    alpha: float = 0.2,
    threshold: float = 0.05,
    mode: str = "constant",
    cval: Union[float, int] = 0,
):
    """
    Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.
    
    Args:
        image (np.ndarray): an image
        ksize (int): The size of the Guassian Kernel. If 0, then ksize is estimated as `round(sigma * 8) + 1`
        sigma (float): Gaussian kernel standard deviation. If 0, then sigma is estimated as `0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`
        alpha (float): visibility of sharpened image
        threshold (float): Value to limit sharpening only for areas with high pixel difference between original image
        mode (str): scipy parameter to determine how the input image is extended during convolution to maintain image shape. Must be one of the following:

            - `reflect` (d c b a | a b c d | d c b a): The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.
            - `constant` (k k k k | a b c d | k k k k): The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
            - `nearest` (a a a a | a b c d | d d d d): The input is extended by replicating the last pixel.
            - `mirror` (d c b | a b c d | c b a): The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.
            - `wrap` (a b c d | a b c d | a b c d): The input is extended by wrapping around to the opposite edge.

            Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html

            Default: `constant`
        cval (int,float): The fill value when mode = `constant`. Default: 0

    Reference:
        https://arxiv.org/pdf/2107.10833.pdf
    """
    input_dtype = image.dtype
    if input_dtype in {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("int32"),
    }:
        image = to_float(image)
    elif input_dtype not in MAX_VALUES_BY_DTYPE.keys():
        raise ValueError(
            "Unexpected dtype {} for UnsharpMask augmentation".format(input_dtype)
        )

    if ksize == 0:
        ksize = round(sigma * 8) + 1

    if sigma == 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    blur_fn = _maybe_process_by_channel(
        ndimage.gaussian_filter,
        sigma=sigma,
        radius=((ksize - 1) // 2,) * 3,
        mode=mode,
        cval=cval,
    )
    blur = blur_fn(image)

    residual = image - blur

    # Do not sharpen noise
    mask = np.abs(residual) > threshold
    mask = mask.astype("float32")

    sharp = image + alpha * residual
    # Avoid color noise artefacts.
    sharp = np.clip(sharp, 0, 1)

    soft_mask = blur_fn(mask)

    output = soft_mask * sharp + (1 - soft_mask) * image
    return from_float(output, dtype=input_dtype)


@preserve_shape
def pixel_dropout(
    image: np.ndarray, drop_mask: np.ndarray, drop_value: Union[float, Sequence[float]]
) -> np.ndarray:
    if isinstance(drop_value, (int, float)) and drop_value == 0:
        drop_values = np.zeros_like(image)
    else:
        drop_values = np.full_like(image, drop_value)  # type: ignore
    return np.where(drop_mask, drop_values, image)


@clipped
@preserve_shape
def spatter(
    img: np.ndarray,
    non_mud: Optional[np.ndarray],
    mud: Optional[np.ndarray],
    rain: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    non_rgb_warning(img)

    coef = MAX_VALUES_BY_DTYPE[img.dtype]
    img = img.astype(np.float32) * (1 / coef)

    if mode == "rain":
        assert rain is not None
        img = img + rain
    elif mode == "mud":
        assert non_mud is not None and mud is not None
        img = img * non_mud + mud
    else:
        raise ValueError("Unsupported spatter mode: " + str(mode))

    return img * 255
