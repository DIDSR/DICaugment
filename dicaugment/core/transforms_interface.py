from __future__ import absolute_import

import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import cv2
import numpy as np

from .serialization import Serializable, get_shortest_class_fullname
from .utils import format_args

__all__ = [
    "to_tuple",
    "BasicTransform",
    "DualTransform",
    "ImageOnlyTransform",
    "NoOp",
    "BoxType",
    "KeypointType",
    "ImageColorType",
    "ScaleFloatType",
    "ScaleIntType",
    "ImageColorType",
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_QUADRATIC",
    "INTER_CUBIC",
    "INTER_QUARTIC",
    "INTER_QUINTIC",
]

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_QUADRATIC = 2
INTER_CUBIC = 3
INTER_QUARTIC = 4
INTER_QUINTIC = 5

NumType = Union[int, float, np.ndarray]
BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
KeypointInternalType = Tuple[float, float, float, float]
KeypointType = Union[KeypointInternalType, Tuple[float, float, float, float, Any]]
ImageColorType = Union[float, Sequence[float]]
DicomType = Dict[str, Any]

ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]

FillValueType = Optional[Union[int, float, Sequence[int], Sequence[float]]]


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple.

    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element

    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        if len(param) != 2:
            raise ValueError("to_tuple expects 1 or 2 values")
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


class BasicTransform(Serializable):
    """
    Abstract Base Class for Transforms. Not intended to be instantiated.
    
    Args:
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    """
    call_backup = None
    interpolation: Any
    fill_value: Any
    mask_fill_value: Any

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets: Dict[str, str] = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Invokes the augmentation pipeline.
        Data passed must be named arguments, for example: aug(image=image)

        Args:
            force_apply(bool): whether to always apply the transformations. Default: False
            **kwargs: keyword arguments for augmentations (e.g, image=image, bboxes=bboxes)

        Returns:
            Dictionary of augmented data
        Raises:
            KeyError: If positional args are passed to this method
        """
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            )
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(
                    key in kwargs for key in self.targets_as_params
                ), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(
                    targets_as_params
                )
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname()
                        + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(
        self, params: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:  # skipcq: PYL-W0613
        """
        Applies the augmentation to each input.

        Args:
            params (dict): keys-value pairs of argument names for the augmentation's `apply` methods
            kwargs (dict): keyword arguments of targets (e.g. 'image', 'bboxes')

        Returns:
            dict of augmented targets
        """
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {
                    k: kwargs[k] for k in self.target_dependence.get(key, [])
                }
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(
        self, flag: bool, save_key: str = "replay"
    ) -> "BasicTransform":
        """
        Enables replays of non-deterministic transforms
        
        Args:
            flag (bool): Whether or not to set the transforms as deterministic
            save_key(str): The dict key where the saved parameters will be found in output. Default: "replay"
        
        Returns:
            self
        """
        assert save_key != "params", "'params' save_key is reserved for internal use"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self) -> str:
        """Returns a string representation of this object"""
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(
            name=self.__class__.__name__, args=format_args(state)
        )

    def _get_target_function(self, key: str) -> Callable:
        """
        Returns the applicable `apply` method for the data type (e.g. 'bboxes' -> apply_to_bboxes())

        Args:
            key (str): a target name (e.g. 'bboxes')

        Returns:
            the respective `apply` method for the key
        """
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Applies the augmentation to an image
        """
        raise NotImplementedError

    def get_params(self) -> Dict:
        """Returns parameters needed for the `apply` methods"""
        return {}

    @property
    def targets(self) -> Dict[str, Callable]:
        """
        Returns the mapping of target to applicable `apply` method.
        (e.g. {'image': self.apply, 'bboxes', self.apply_to_bboxes})
        """
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Adds additional parameters that are defined at a per instance level

        Args:
            params (dict): keys-value pairs of argument names for the augmentation's `apply` methods
            kwargs (dict): keyword arguments of targets (e.g. 'image', 'bboxes')
        """
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        params.update(
            {
                "cols": kwargs["image"].shape[1],
                "rows": kwargs["image"].shape[0],
                "slices": kwargs["image"].shape[2],
            }
        )
        return params

    @property
    def target_dependence(self) -> Dict:
        """An unused alternate form of the `get_parameters` and `get_params_dependent_on_targets`"""
        return {}

    def add_targets(self, additional_targets: Dict[str, str]):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'

        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self) -> List[str]:
        """Returns a list of target names (e.g. 'image') that are needed as a parameter input
        to other `apply` methods (e.g. apply_to_bboxes(..., image = image))
        """
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class "
            + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls) -> str:
        """Returns shortened submodule path name"""
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls):
        """Returns whether the class is serializable"""
        return True

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self) -> Dict[str, Any]:
        """Returns base initialization argument names used in every transform"""
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self) -> Dict[str, Any]:
        """Returns initialization arguments (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1' : 1, 'arg2': 2))"""
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self) -> Dict[str, Any]:
        """Returns a serializable representation of object"""
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self) -> Dict[str, Any]:
        """Returns a serializable representation of object with a unique integer identifier for the object"""
        d = self._to_dict()
        d["id"] = id(self)
        return d


class DualTransform(BasicTransform):
    """
    Transform for tasks where bboxes, keypoints, and masks could be altered.
    
    Args:
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    """

    @property
    def targets(self) -> Dict[str, Callable]:
        """
        Returns the mapping of target to applicable `apply` method.
        (e.g. {'image': self.apply, 'bboxes', self.apply_to_bboxes})
        """
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "dicom": self.apply_to_dicom,
        }

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """
        Applies the augmentation to a bbox

        Args:
            bbox (BoxInternalType): an internal bbox representation

        Returns:
            an augmented internal bbox representation
        """
        raise NotImplementedError(
            "Method apply_to_bbox is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """
        Applies the augmentation to a keypoint

        Args:
            keypoint (keypointInternalType): an internal keypoint representation

        Returns:
            an augmented internal keypoint representation
        """
        raise NotImplementedError(
            "Method apply_to_keypoint is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], **params) -> List[BoxType]:
        """Applies the augmentation to a sequence of bbox types. See `apply_to_bbox`"""
        return [tuple(self.apply_to_bbox(tuple(bbox[:6]), **params)) + tuple(bbox[6:]) for bbox in bboxes]  # type: ignore

    def apply_to_keypoints(
        self, keypoints: Sequence[KeypointType], **params
    ) -> List[KeypointType]:
        """Applies the augmentation to a sequence of keypoint types. See `apply_to_keypoints`"""
        return [  # type: ignore
            self.apply_to_keypoint(tuple(keypoint[:5]), **params) + tuple(keypoint[5:])  # type: ignore
            for keypoint in keypoints
        ]

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the augmentation to a mask and forces INTER_NEAREST interpolation"""
        return self.apply(
            img,
            **{
                k: INTER_NEAREST if k == "interpolation" else v
                for k, v in params.items()
            }
        )

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params) -> List[np.ndarray]:
        """Applies the augmentation to a sequence of mask types. See `apply_to_mask`"""
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def apply_to_dicom(self, dicom: DicomType, **params) -> DicomType:
        """Applies the augmentation to a dicom type"""
        return dicom


class ImageOnlyTransform(BasicTransform):
    """
    Transform applied to image only. bboxes, keypoints, and masks are unnaffected.
    
    Args:
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    """

    @property
    def targets(self) -> Dict[str, Callable]:
        """
        Returns the mapping of target to applicable `apply` method.
        (e.g. {'image': self.apply, 'bboxes', self.apply_to_bboxes})
        """
        return {"image": self.apply}


class NoOp(DualTransform):
    """
    Does nothing. Applies no augmentations
    
    Args:
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    """

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, **params
    ) -> KeypointInternalType:
        """Returns keypoint"""
        return keypoint

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        """Returns bbox"""
        return bbox

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Returns image"""
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """returns mask"""
        return img

    def get_transform_init_args_names(self) -> Tuple:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return ()
