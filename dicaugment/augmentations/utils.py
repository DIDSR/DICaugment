from functools import wraps
from typing import Callable, Union

import cv2
import numpy as np
import os
import pydicom as pdm
from typing_extensions import Concatenate, ParamSpec

from dicaugment.core.keypoints_utils import angle_to_2pi_range
from dicaugment.core.transforms_interface import KeypointInternalType

__all__ = [
    "read_dcm_image",
    "MAX_VALUES_BY_DTYPE",
    "MIN_VALUES_BY_DTYPE",
    "NPDTYPE_TO_OPENCV_DTYPE",
    "NPDTYPE_TO_OPENCV_DTYPE",
    "clipped",
    "angle_2pi_range",
    "clip",
    "preserve_shape",
    "preserve_channel_dim",
    "ensure_contiguous",
    "is_rgb_image",
    "is_grayscale_image",
    "is_multispectral_image",
    "get_num_channels",
    "non_rgb_warning",
    "_maybe_process_in_chunks",
    "_maybe_process_by_channel",
]

P = ParamSpec("P")

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
    np.dtype("int16"): 32767,
    np.dtype("int32"): 2147483647,
    np.dtype("float64"): np.finfo(np.float64).max,
}

MIN_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 0,
    np.dtype("uint16"): 0,
    np.dtype("uint32"): 0,
    np.dtype("float32"): 0.0,
    np.dtype("int16"): -32768,
    np.dtype("int32"): -2147483648,
    np.dtype("float64"): np.finfo(np.float64).min,
}


NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}

SCIPY_MODE_TO_NUMPY_MODE = {
    "reflect": "symmetric",
    "constant": "constant",
    "nearest": "edge",
    "mirror": "reflect",
    "wrap": "wrap",
}


def read_dcm_image(path: str, include_header: bool = True, ends_with: str = ""):
    """
    Reads in an alphabetically sorted series of dcm file types stored in a directory as a `np.ndarray` and optionally a dicom header in a `dict` format.

    Args:
        path (str): The filepath to the directory that stores the dcm files.
        include_header (bool): Whether to return the dicom header metadata associated with the scan.
            Default: True
        ends_with (str): If empty string, then all files in directory will be processed. If multiple file types are within the directory, you may filter the results by setting `ends_with=".dcm"`
            Default: ""

    Note:
        `DICOM` object types are dictionaries with the following keys:
            `PixelSpaxing` (tuple)
                The space in mm between pixels for both height and width of a slice, respectively
            `RescaleIntercept` (float)
                The value to add to each pixel of the scan after scaling with `RescaleSlope` to turn the pixel values of the scan into Hounsfield Units (HU)
            `RescaleSlope` (float)
                The value to multiply each pixel of the scan by before adding `RescaleIntercept` to turn the pixel values of the scan into Hounsfield Units (HU)
            `ConvolutionKernel` (str)
                A label describing the convolution kernel or algorithm used to reconstruct the data
            `XRayTubeCurrent` (int)
                X-Ray Tube Current in mA.

        See example below:


    .. code-block:: python

        dicom = {
            "PixelSpacing" : (0.5, 0.5),
            "RescaleIntercept" : -1024.0,
            "RescaleSlope" : 1.0,
            "ConvolutionKernel" : 'STANDARD',
            "XRayTubeCurrent" : 160
        }
    """

    if not os.path.isdir(path):
        raise OSError("{} is not a valid directory".format(path))

    img = None

    for file in sorted(os.listdir(path)):
        if not file.endswith(ends_with):
            continue

        fp = os.path.join(path, file)
        obj = pdm.dcmread(fp)
        dcm = np.expand_dims(obj.pixel_array, axis=2).astype(np.int16)

        if img is None:
            img = dcm
            dicom = {
                "PixelSpacing": tuple(map(float, obj.PixelSpacing)),
                "RescaleIntercept": float(obj.RescaleIntercept),
                "RescaleSlope": float(obj.RescaleSlope),
                "ConvolutionKernel": obj.ConvolutionKernel,
                "XRayTubeCurrent": int(obj.XRayTubeCurrent),
            }
        else:
            img = np.concatenate([img, dcm], axis=2)

    if include_header:
        return img, dicom

    return img


def clipped(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray]
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Decorator method that clips an image to it's specified dtype minimums and maximums"""
    @wraps(func)
    def wrapped_function(
        img: np.ndarray, *args: P.args, **kwargs: P.kwargs
    ) -> np.ndarray:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        minval = MIN_VALUES_BY_DTYPE.get(dtype, 0.0)
        return clip(func(img, *args, **kwargs), dtype, minval, maxval)

    return wrapped_function


def clip(img: np.ndarray, dtype: np.dtype, minval: float, maxval: float) -> np.ndarray:
    """Clips an image by a minimum and maximum value, then casts to dtype"""
    return np.clip(img, minval, maxval).astype(dtype)

def angle_2pi_range(
    func: Callable[Concatenate[KeypointInternalType, P], KeypointInternalType]
) -> Callable[Concatenate[KeypointInternalType, P], KeypointInternalType]:
    """Decorator method that keeps keypoints angles in the range of [0, 2pi]"""
    @wraps(func)
    def wrapped_function(
        keypoint: KeypointInternalType, *args: P.args, **kwargs: P.kwargs
    ) -> KeypointInternalType:
        (x, y, z, a, s) = func(keypoint, *args, **kwargs)[:5]
        return (x, y, z, angle_to_2pi_range(a), s)

    return wrapped_function

def preserve_shape(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray]
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Decorators that preserves shape of the image"""

    @wraps(func)
    def wrapped_function(
        img: np.ndarray, *args: P.args, **kwargs: P.kwargs
    ) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function

def preserve_channel_dim(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray]
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Decorator that preserves a dummy channel dim."""

    @wraps(func)
    def wrapped_function(
        img: np.ndarray, *args: P.args, **kwargs: P.kwargs
    ) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 4 and shape[-1] == 1 and len(result.shape) == 3:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def ensure_contiguous(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray]
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Decorator that ensures input img is contiguous."""

    @wraps(func)
    def wrapped_function(
        img: np.ndarray, *args: P.args, **kwargs: P.kwargs
    ) -> np.ndarray:
        img = np.require(img, requirements=["C_CONTIGUOUS"])
        result = func(img, *args, **kwargs)
        return result

    return wrapped_function


def is_rgb_image(image: np.ndarray) -> bool:
    """Returns whether image fits the criteria for a volumetric rgb image"""
    return (
        len(image.shape) == 4
        and image.shape[-1] == 3
        and image.dtype in {np.dtype("uint8"), np.dtype("float32")}
    )


def is_grayscale_image(image: np.ndarray) -> bool:
    """Returns whether image fits the criteria for a volumetric grayscale image"""
    return (len(image.shape) == 3) or (len(image.shape) == 4 and image.shape[-1] == 1)


def is_uint8_or_float32(image: np.ndarray) -> bool:
    """Returns whether image is type `uint8` or `float32`"""
    return image.dtype in {np.dtype("uint8"), np.dtype("float32")}


def is_multispectral_image(image: np.ndarray) -> bool:
    """Returns whether image fits the criteria for a volumetric multispectral image"""
    return len(image.shape) == 4 and image.shape[-1] not in [1, 3]


def get_num_channels(image: np.ndarray) -> int:
    """Returns number of channels in image"""
    return image.shape[3] if len(image.shape) == 4 else 1


def non_rgb_warning(image: np.ndarray) -> None:
    """Warns user if image is not an RGB image"""
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_grayscale_image(image):
            message += "\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        if is_multispectral_image(
            image
        ):  # Any image with a number of channels other than 1 and 3
            message += (
                "\nThis transformation cannot be applied to multi-spectral images"
            )

        raise ValueError(message)


def _maybe_process_in_chunks(
    process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray], **kwargs
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def _maybe_process_by_channel(
    process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray], **kwargs
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap OpenCV or Scipy function to enable processing channeled images of any length.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g scipy.ndimage.zoom).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = get_num_channels(img)
        if num_channels > 1 or len(img.shape) > 3:
            chunks = []
            for i in range(num_channels):
                chunks.append(process_fn(img[..., i], **kwargs))

            img = np.stack(chunks, axis=3)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn
