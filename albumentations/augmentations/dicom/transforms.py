
from typing import Dict, Optional, Sequence, Tuple, Union, Callable

import numpy as np
from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    ImageOnlyTransform,
    ImageColorType,
    KeypointInternalType,
    ScaleFloatType,
    DicomType,
    to_tuple,
    INTER_LINEAR,
    INTER_NEAREST
)

from . import functional as F
from ..geometric import functional as FGeometric


__all__ = [
    "RescaleSlopeIntercept",
    "SetPixelSpacing"
]

class RescaleSlopeIntercept(ImageOnlyTransform):
    """
    Rescales img to Hounsfields Units (HU) using the `(0028, 1053) Rescale Slope` and `(0028, 1052) Rescale Intercept` values from a dicom header. Long Doc String......
    """

    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, slope: float, intercept: float, **params) -> np.ndarray:
        return F.rescale_slope_intercept(img, slope, intercept)
    
    def apply_to_dicom(self, dicom, **params):
        return F.reset_dicom_slope_intercept(dicom)
    
    def get_params_dependent_on_targets(self, params):
        slope = params["dicom"]["RescaleSlope"]
        intercept = params["dicom"]["RescaleIntercept"]
        return {"slope": slope, "intercept": intercept}
    
    @property
    def targets_as_params(self):
        return ["dicom"]
    
    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "dicom": self.apply_to_dicom
            }

    # def get_transform_init_args_names(self) -> Tuple[str, ...]:
    #     return ("blur_limit", "by_slice", "mode", "cval")



class SetPixelSpacing(DualTransform):
    """
    Resize an image so that the `(0028, 0030) Pixel Spacing` values and optionally the `(0018, 0050) Slice Thickness` 
    value of the dicom header are equal to `space_x`, `space_y`, and `space_z`, respectively
    
    Args:
        space_x (float): desired pixel spacing in the width dimension.  Default: 1.0
        space_y (float): desired pixel spacing in the height dimension. Default: 1.0
        space_z (float): desired pixel spacing in the depth dimension.  Default: 1.0
        set_thickness (bool): Whether to alter the slice thickness of the image. If False, then space_z is ignored. Default: True
        interpolation (int): scipy interpolation method (e.g. albumenations3d.INTER_NEAREST). Default: albumentations3d.INTER_LINEAR
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, space_x: float = 1.0, space_y: float = 1.0, space_z: float = 1.0, set_thickness: bool = True, interpolation=INTER_LINEAR, always_apply=False, p=1):
        super(SetPixelSpacing, self).__init__(always_apply, p)
        self.space_x = space_x
        self.space_y = space_y
        self.space_z = space_z
        self.set_thickness = set_thickness
        self.interpolation = interpolation

        assert space_x > 0, "Pixel Spaxing must be nonegative for argument space_x, got {}".format(space_x)
        assert space_y > 0, "Pixel Spaxing must be nonegative for argument space_y, got {}".format(space_y)
        assert space_z > 0, "Pixel Spaxing must be nonegative for argument space_z, got {}".format(space_z)

    def apply(self, img, interpolation=INTER_LINEAR, scale_x: float = 1.0, scale_y: float = 1.0, scale_z: float = 1.0, **params):
        height, width, depth = img.shape[:3]
        return FGeometric.resize(img, height=height*scale_y, width=width*scale_x, depth=depth*scale_y, interpolation=interpolation)
    
    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox
    
    def apply_to_keypoint(self, keypoint, scale_x: float, scale_y:float, scale_z:float, **params):
        return FGeometric.keypoint_scale(keypoint, scale_x, scale_y, scale_z)
    
    def apply_to_dicom(self, dicom: DicomType, scale_x: float, scale_y:float, scale_z:float, **params) -> DicomType:
        return F.dicom_scale(dicom, scale_x, scale_y, scale_z)

    def get_params_dependent_on_targets(self, params):
        y, x = params["dicom"]["PixelSpacing"]
        z = params["dicom"]["SliceThickness"]
        scale_x = self.space_x / x
        scale_y = self.space_y / y
        scale_z = self.space_z / z if self.set_thickness else 1
        return {"scale_x": scale_x, "scale_y": scale_y, "scale_z": scale_z}
    
    @property
    def targets_as_params(self):
        return ["dicom"]
    
    def get_transform_init_args_names(self):
        return ("space_x", "space_y", "space_z", "set_thickness", "interpolation")