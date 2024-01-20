import numpy as np
from ...core.transforms_interface import DicomType
import pkg_resources
from typing import Tuple

__all__ = [
    "rescale_slope_intercept",
    "reset_dicom_slope_intercept",
    "dicom_scale",
    "transpose_dicom",
]


def rescale_slope_intercept(
    img: np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    """
    Scales and offsets an image's pixel values using the formula `img = (img * slope) + intercept`

    Args:
        img (np.ndarray): an image
        slope (float): the factor to scale the pixel values by
        intercept (float): the value to offset the pixel values by
    """
    img = img.astype(np.int16)
    img *= slope
    img += intercept
    return img


def reset_dicom_slope_intercept(dicom: DicomType) -> DicomType:
    """
    Sets the `RescaleSlope` and `RescaleIntercept` keys of a Dicom Object to 1 and 0, respectively.
    
    Args:
        dicom (DicomType): a Dicom object
    """
    res = {}
    for k, v in dicom.items():
        res[k] = v
    res["RescaleSlope"] = 1
    res["RescaleIntercept"] = 0
    return res


def dicom_scale(dicom: DicomType, scale_x: float, scale_y: float) -> DicomType:
    """
    Scales the `PixelSpacing` of a Dicom Object
    
    Args:
        dicom (DicomType): a Dicom object
        scale_x (float): factor to scale the PixelSpacing in the x dimension
        scale_y (float): factor to scale the PixelSpacing in the y dimension
    """
    y, x = dicom["PixelSpacing"]
    x *= scale_x
    y *= scale_y

    res = {}
    for k, v in dicom.items():
        res[k] = v

    res["PixelSpacing"] = (y, x)
    return res


def transpose_dicom(dicom: DicomType) -> DicomType:
    """
    Transposes the `PixelSpacing` of a Dicom Object
    
    Args:
        dicom (DicomType): a Dicom object
    """
    y, x = dicom["PixelSpacing"]

    res = {}
    for k, v in dicom.items():
        res[k] = v

    res["PixelSpacing"] = (x, y)
    return res


def _load_kernel(kname: str = "STANDARD") -> np.ndarray:
    """
    Return a numpy array of the specified kernel name

    """
    fname = pkg_resources.resource_filename(
        __name__, "data/kernels/{}.npy".format(kname.lower())
    )

    try:
        return np.load(fname)
    except Exception as e:
        raise ValueError(
            "{}\nCould not find kernel with name {}.npy".format(e, kname.lower())
        )


def _generate_NPS_noise(NPS: np.ndarray) -> np.ndarray:
    n = np.random.random(NPS.shape).astype("float64")
    phase_shift = np.sqrt(NPS) * (np.cos(2 * np.pi * n) + 1j * np.sin(2 * np.pi * n))
    ift = np.fft.ifftn(phase_shift, axes=tuple(range(NPS.ndim)))
    noise = np.real(ift) + np.imag(ift)
    coef = 1 / np.std(noise)
    return coef * noise


def _nps_radial_to_cartesian(
    rad_nps: np.ndarray, shape: Tuple, x_step: float, y_step: float
) -> np.ndarray:
    nNPS = np.zeros(shape=shape)
    freq_row = np.fft.fftfreq(n=shape[1], d=x_step)
    freq_col = np.fft.fftfreq(n=shape[0], d=y_step)

    spatial_freq = rad_nps["Spatial_frequency"]
    radial_val = rad_nps["nNPS"]

    for i in range(shape[1]):
        for j in range(shape[0]):
            spatial_val = np.sqrt(freq_row[i] ** 2 + freq_col[j] ** 2)

            idx = min(
                range(len(spatial_freq) - 1),
                key=lambda x: abs(spatial_freq[x] - spatial_val),
            )
            nNPS[j, i] = radial_val[idx]

    nNPS[nNPS < 0] = 0
    return nNPS


def _noise_to_3d(nps: np.ndarray, slices: int, magnitude: int) -> np.ndarray:
    return (
        np.repeat(_generate_NPS_noise(nps)[..., np.newaxis], slices, axis=2) * magnitude
    )


def add_noise_nps(
    img: np.ndarray, kernel: str, x_step: float, y_step: float, magnitude: int
) -> np.ndarray:
    height, width, depth = img.shape[:3]
    rad_nps = _load_kernel(kernel)
    nps = _nps_radial_to_cartesian(
        rad_nps=rad_nps, shape=(height, width), x_step=x_step, y_step=y_step
    )
    nps3d = _noise_to_3d(nps=nps, slices=depth, magnitude=magnitude)
    out = img.copy()
    return out + nps3d.astype(np.int16)
