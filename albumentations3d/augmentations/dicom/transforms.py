
from typing import Dict, Optional, Sequence, Tuple, Union, Callable, Any, List

import numpy as np
import random
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
    "SetPixelSpacing",
    "NPSNoise"
]

class RescaleSlopeIntercept(ImageOnlyTransform):
    """
    Rescales img to Hounsfields Units (HU) using the `(0028, 1053) Rescale Slope` and `(0028, 1052) Rescale Intercept` values from a dicom header.
    This will return the image with data type `np.int16`.
    
    Args:
        None

    Targets:
        image, dicom

    Image types:
        int16, uint16
    
    Note:
        This transformation requires the use a DICOM header object. See `albumentations3d.read_dcm_image()` for full syntax.
        .. code-block:: python

            import albumentations3d as A
            image, dicom = A.read_dcm_image(path='path/to/dcm/folder/', return_header=True)
            aug = A.Compose([A.RescaleSlopeIntercept()])
            result = aug(image=image, dicom=dicom)
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

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()



class SetPixelSpacing(DualTransform):
    """
    Resize an image so that the `(0028, 0030) Pixel Spacing` values of the dicom header are equal to `space_x` and `space_y` respectively
    
    Args:
        space_x (float): desired pixel spacing in the width dimension.  Default: 1.0
        space_y (float): desired pixel spacing in the height dimension. Default: 1.0
        interpolation (int): scipy interpolation method (e.g. albumenations3d.INTER_NEAREST). Default: albumentations3d.INTER_LINEAR
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, dicom, mask, bboxes, keypoints

    Image types:
        uint8, uint16, int16, float32
    
    Note:
        This transformation requires the use a DICOM header object. See `albumentations3d.read_dcm_image()` for full syntax.
    
    Example:
        .. code-block:: python

            import albumentations3d as A
            image, dicom = A.read_dcm_image(path='path/to/dcm/folder/', return_header=True)
            aug = A.Compose([A.SetPixelSpacing(space_x=0.5, space_y=0.5)])
            result = aug(image=image, dicom=dicom)
    """

    def __init__(self, space_x: float = 1.0, space_y: float = 1.0, interpolation=INTER_LINEAR, always_apply=False, p=1):
        super(SetPixelSpacing, self).__init__(always_apply, p)
        self.space_x = space_x
        self.space_y = space_y
        self.interpolation = interpolation

        assert space_x > 0, "Pixel Spaxing must be nonegative for argument space_x, got {}".format(space_x)
        assert space_y > 0, "Pixel Spaxing must be nonegative for argument space_y, got {}".format(space_y)

    def apply(self, img: np.ndarray, interpolation: int = INTER_LINEAR, scale_x: float = 1.0, scale_y: float = 1.0, **params) -> np.ndarray:
        height, width, depth = img.shape[:3]
        return FGeometric.resize(img, height=height*scale_y, width=width*scale_x, depth=depth, interpolation=interpolation)
    
    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox
    
    def apply_to_keypoint(self, keypoint: KeypointInternalType, scale_x: float, scale_y:float, scale_z:float, **params) -> KeypointInternalType:
        return FGeometric.keypoint_scale(keypoint, scale_x, scale_y, 1)
    
    def apply_to_dicom(self, dicom: DicomType, scale_x: float, scale_y:float, **params) -> DicomType:
        return F.dicom_scale(dicom, scale_x, scale_y)

    def get_params_dependent_on_targets(self, params: Dict[str,Any]) -> Dict[str,Any]:
        y, x = params["dicom"]["PixelSpacing"]
        scale_x = self.space_x / x
        scale_y = self.space_y / y
        return {"scale_x": scale_x, "scale_y": scale_y}
    
    @property
    def targets_as_params(self) -> List[str]:
        return ["dicom"]
    
    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("space_x", "space_y", "interpolation")
    

class NPSNoise(ImageOnlyTransform):
    """
    Insert random image noise based on the `(0018,1210) Convolution Kernel` type of the dicom header.

    Args:
        magnitude ((int, int) or int): scaling magnitude range of noise. If magnitude is a single integer value, the
            range will be (1, magnitude). Default: (50, 150).
        sample_tube_current (bool): If True, then magnitude is ignored and the magnitude is sampled from the range (0, 500 - `(0018,1151) X-Ray Tube Current`)
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, dicom

    Image types:
        int16

    Note:
        This transformation requires the use a DICOM header object. See `albumentations3d.read_dcm_image()` for full syntax.
        .. code-block:: python

            import albumentations3d as A
            image, dicom = A.read_dcm_image(path='path/to/dcm/folder/', return_header=True)
            aug = A.Compose([A.NPSNoise()])
            result = aug(image=image, dicom=dicom)
    """

    def __init__(self, magnitude: int = (50, 150), sample_tube_current: bool = False, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.magnitude = to_tuple(magnitude, low= 0)
        self.sample_tube_current = sample_tube_current

        assert magnitude[0] >= 0, "magnitude range must be nonnegative, got {}".format(magnitude)

    def apply(self, img: np.ndarray, kernel: str = 'STANDARD', x_step: float = 0.5, y_step: float = 0.5, magnitude: int = 1, **params):
        return F.add_noise_nps(img, kernel=kernel, x_step=x_step, y_step=y_step, magnitude=magnitude)
        
    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kernel = params["dicom"]["ConvolutionKernel"]
        y, x = params["dicom"]["PixelSpacing"]
        if self.sample_tube_current:
            return {"kernel": kernel, "x_step": x, "y_step": y, "magnitude": random.uniform(0, 500 - params["dicom"]["XRayTubeCurrent"])}
        return {"kernel": kernel, "x_step": x, "y_step": y}
    
    def get_params(self) -> Dict:
        return {"magnitude": random.uniform(self.magnitude[0], self.magnitude[1])}

    @property
    def targets_as_params(self):
        return ["dicom"]
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("magnitude", "sample_tube_current")