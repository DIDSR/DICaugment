from __future__ import absolute_import

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .serialization import Serializable


def get_shape(img: Union[np.ndarray, 'torch.tensor']) -> Tuple[int, int, int]: # noqa: F821
    """
    Returns the shape of an image depending on if it is a numpy array or torch tensor
    
    Args:
        img (arraylike):  A numpy array or torch tensor

    Returns:
        The shape of the image
    
    Raises:
        RuntimeError: if image is not a numpy array or torch tensor
    """
    if isinstance(img, np.ndarray):
        if img.ndim not in {3, 4}:
            raise ValueError(
                f"Albumenatations3D expected numpy.ndarray or torch.Tensor of shape (H,W,D) or (H,W,D,C). Got: {img.shape}"
            )
        rows, cols, slices = img.shape[:3]
        return rows, cols, slices

    try:
        import torch

        if torch.is_tensor(img):
            if img.ndim not in {3, 4}:
                raise ValueError(
                    f"Albumenatations3D expected numpy.ndarray or torch.Tensor of shape (H,W,D) or (H,W,D,C). Got: {img.shape}"
                )
            if img.ndim == 3:
                slices, rows, cols = img.shape[:3]
            else:
                slices, rows, cols = img.shape[1:]
            return rows, cols, slices
    except ImportError:
        pass

    raise RuntimeError(
        f"Dicaugment supports only numpy.ndarray and torch.Tensor data type for image. Got: {type(img)}"
    )


def format_args(args_dict: Dict) -> str:
    """Returns a string representation of an arguments dictionary"""
    formatted_args = []
    for k, v in args_dict.items():
        if isinstance(v, str):
            v = f"'{v}'"
        formatted_args.append(f"{k}={v}")
    return ", ".join(formatted_args)


class Params(Serializable, ABC):
    """
    Abstract Base Class for parameters

    Args:
        format (str): the format that a parameters should be interpreted as. Formats defined in subclasses
        label_fields (list): list of fields that are joined with the parameters, e.g labels.
    """
    def __init__(self, format: str, label_fields: Optional[Sequence[str]] = None):
        self.format = format
        self.label_fields = label_fields

    def _to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "label_fields": self.label_fields}


class DataProcessor(ABC):
    """
    Abstract Base Class for processors

    Args:
        params (Params): a parameter object
        additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
    """
    def __init__(
        self, params: Params, additional_targets: Optional[Dict[str, str]] = None
    ):
        self.params = params
        self.data_fields = [self.default_data_name]
        if additional_targets is not None:
            for k, v in additional_targets.items():
                if v == self.default_data_name:
                    self.data_fields.append(k)

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        """Returns the default data name for class (e.g. 'image')"""
        raise NotImplementedError

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        """Raises a ValueError if input data is not in the expected format
        (e.g. `label_fields` does not match up with values with params dict)"""
        pass

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        """Unused method to check if params are valid for DualIAATransforms"""
        pass

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms params from their respective internal type to the type specified in Params.format
        
        Args:
            data (dict): A dictionary of targets (e.g. {'image': np.ndarray(...), ...})
        
        Returns:
            The input `data` dictionary but with each target transformed back from the internal dicaugment type
        """
        rows, cols, slices = get_shape(data["image"])

        for data_name in self.data_fields:
            data[data_name] = self.filter(data[data_name], rows, cols, slices)
            data[data_name] = self.check_and_convert(
                data[data_name], rows, cols, slices, direction="from"
            )

        data = self.remove_label_fields_from_data(data)
        return data

    def preprocess(self, data: Dict[str, Any]) -> None:
        """
        Transforms params from their type specified in Params.format to their respective internal type
        
        Args:
            data (dict): A dictionary of targets (e.g. {'image': np.ndarray(...), ...})
        
        Returns:
            The input `data` dictionary but with each target transformed to the internal dicaugment type
        """
        data = self.add_label_fields_to_data(data)

        rows, cols, slices = get_shape(data["image"])
        for data_name in self.data_fields:
            data[data_name] = self.check_and_convert(
                data[data_name], rows, cols, slices, direction="to"
            )

    def check_and_convert(
        self, data: Sequence, rows: int, cols: int, slices: int, direction: str = "to"
    ) -> Sequence:
        """
        Converts data to or from the `dicaugment_3d` format
        
        Args:
            data (Sequence): a target (e.g. a bbox or keypoint)
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image
            direction (str): Whether to transform 'to' or 'from' the `dicaugment_3d` format

        Returns:
            data converted to or from the `dicaugment_3d` format

        Raises:
            ValueError: if `direction` is not `to` or `from`
        """
        if self.params.format == "dicaugment_3d":
            self.check(data, rows, cols, slices)
            return data

        if direction == "to":
            return self.convert_to_dicaugment(data, rows, cols, slices)
        elif direction == "from":
            return self.convert_from_dicaugment(data, rows, cols, slices)
        else:
            raise ValueError(
                f"Invalid direction. Must be `to` or `from`. Got `{direction}`"
            )

    @abstractmethod
    def filter(self, data: Sequence, rows: int, cols: int, slices: int) -> Sequence:
        """Wrapper method to invoke filter methods for subclasses"""
        pass

    @abstractmethod
    def check(self, data: Sequence, rows: int, cols: int, slices: int) -> None:
        """Wrapper method to invoke check methods for subclasses"""
        pass

    @abstractmethod
    def convert_to_dicaugment(
        self, data: Sequence, rows: int, cols: int, slices: int
    ) -> Sequence:
        """
        Converts data to the `dicaugment_3d` format
        
        Args:
            data (Sequence): a target (e.g. a bbox or keypoint)
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image

        Returns:
            data converted to the `dicaugment_3d` format
        """
        pass

    @abstractmethod
    def convert_from_dicaugment(
        self, data: Sequence, rows: int, cols: int, slices: int
    ) -> Sequence:
        """
        Converts data from the `dicaugment_3d` format
        
        Args:
            data (Sequence): a target (e.g. a bbox or keypoint)
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image

        Returns:
            data converted from the `dicaugment_3d` format
        """
        pass

    def add_label_fields_to_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Adds label fields to data"""
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            for field in self.params.label_fields:
                assert len(data[data_name]) == len(data[field])
                data_with_added_field = []
                for d, field_value in zip(data[data_name], data[field]):
                    data_with_added_field.append(list(d) + [field_value])
                data[data_name] = data_with_added_field
        return data

    def remove_label_fields_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Removes label fields to data"""
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            label_fields_len = len(self.params.label_fields)
            for idx, field in enumerate(self.params.label_fields):
                field_values = []
                for bbox in data[data_name]:
                    field_values.append(bbox[-label_fields_len + idx])
                data[field] = field_values
            if label_fields_len:
                data[data_name] = [d[:-label_fields_len] for d in data[data_name]]
        return data
