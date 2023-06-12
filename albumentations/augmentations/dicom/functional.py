import numpy as np
from ...core.transforms_interface import DicomType

__all__ = [
    "rescale_slope_intercept",
    "reset_dicom_slope_intercept",
    "dicom_scale",
    "transpose_dicom"
]


def rescale_slope_intercept(img, slope, intercept) -> np.ndarray:
    img = img.astype(np.int16)
    img *= slope
    img += intercept
    return img

def reset_dicom_slope_intercept(dicom) -> DicomType:
    dicom["RescaleSlope"] = 1
    dicom["RescaleIntercept"] = 0
    return dicom

def dicom_scale(dicom: DicomType, scale_x: float, scale_y: float, scale_z: float) -> DicomType:
    y, x = dicom["PixelSpacing"]
    z = dicom["SliceThickness"]
    x *= scale_x
    y *= scale_y
    z *= scale_z
    dicom["PixelSpacing"] = (y, x)
    dicom["SliceThickness"] = z
    return dicom

def transpose_dicom(dicom: DicomType) -> DicomType:
    y, x = dicom["PixelSpacing"]
    dicom["PixelSpacing"] = (x, y)
    return dicom