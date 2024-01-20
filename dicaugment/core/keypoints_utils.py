from __future__ import division

import math
import typing
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import DataProcessor, Params

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_dicaugment",
    "convert_keypoints_to_dicaugment",
    "filter_keypoints",
    "KeypointsProcessor",
    "KeypointParams",
]

keypoint_formats = {"xyz", "zyx", "xyza", "xyzs", "xyzas", "xyzsa"}


def angle_to_2pi_range(angle: float) -> float:
    two_pi = 2 * math.pi
    return angle % two_pi


class KeypointParams(Params):
    """
    Parameters of keypoints

    Args:
        format (str): format of keypoints. Should be 'xyz', 'zyx', 'xyza', 'xyzs', 'xyzas', 'xyzsa'.

            x - X coordinate,

            y - Y coordinate,

            z - Z coordinate,

            s - Keypoint scale

            a - Keypoint planar orientation in radians or degrees (depending on KeypointParams.angle_in_degrees)

        label_fields (list): list of fields that are joined with keypoints, e.g labels.
            Should be same type as keypoints.
        remove_invisible (bool): to remove invisible points after transform or not
        angle_in_degrees (bool): planar angle in degrees or radians in 'xyza', 'xyzas', 'xyzsa' keypoints
        check_each_transform (bool): if `True`, then keypoints will be checked after each dual transform.
            Default: `True`
    """

    def __init__(
        self,
        format: str,  # skipcq: PYL-W0622
        label_fields: Optional[Sequence[str]] = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
    ):
        super(KeypointParams, self).__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

    def _to_dict(self) -> Dict[str, Any]:
        data = super(KeypointParams, self)._to_dict()
        data.update(
            {
                "remove_invisible": self.remove_invisible,
                "angle_in_degrees": self.angle_in_degrees,
                "check_each_transform": self.check_each_transform,
            }
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        """Returns whether the class is serializable"""
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        """Returns class name"""
        return "KeypointParams"


class KeypointsProcessor(DataProcessor):
    """
    Processor Class for Keypoints

    Args:
        params (KeypointParams): An instance of KeypointParams
        additional_targets (dict): keys - new target name, values - old target name. ex: {'keypoints2': 'keypoints'}
    """
    def __init__(
        self,
        params: KeypointParams,
        additional_targets: Optional[Dict[str, str]] = None,
    ):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        """Returns the default data name for class (e.g. 'image')"""
        return "keypoints"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        """Raises a ValueError if input data is not in the expected format
        (e.g. `label_fields` does not match up with values with params dict)"""
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same names as params in "
                    "'keypoint_params' dict"
                )

    def filter(
        self, data: Sequence[Sequence], rows: int, cols: int, slices: int
    ) -> Sequence[Sequence]:
        """
        Wrapper method that invokes `filter_keypoints`. 
        Filters out keypoints that are no longer within the bounds of the image
    
        Args:
            data (Sequence): A sequence of keypoint objects
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image
        
        Returns:
            keypoints
        """
        self.params: KeypointParams
        return filter_keypoints(
            data, rows, cols, slices, remove_invisible=self.params.remove_invisible
        )

    def check(
        self, data: Sequence[Sequence], rows: int, cols: int, slices: int
    ) -> None:
        """
        Wrapper method that invokes `check_keypoints`.
        Checks if keypoint coordinates are less than image shapes or not in correct angle range.
        
        Args:
            data (Sequence): A sequence of keypoint objects
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image
        """
        check_keypoints(data, rows, cols, slices)

    def convert_from_dicaugment(
        self, data: Sequence[Sequence], rows: int, cols: int, slices: int
    ) -> List[Tuple]:
        """
        Converts keypoints from the `dicaugment_3d` format
        
        Args:
            data (Sequence): keypoints
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image

        Returns:
            data converted from the `dicaugment_3d` format
        """
        params = self.params
        return convert_keypoints_from_dicaugment(
            data,
            params.format,
            rows,
            cols,
            slices,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )

    def convert_to_dicaugment(
        self, data: Sequence[Sequence], rows: int, cols: int, slices: int
    ) -> List[Tuple]:
        """
        Converts keypoints to the `dicaugment_3d` format
        
        Args:
            data (Sequence): keypoints
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image

        Returns:
            data converted to the `dicaugment_3d` format
        """
        params = self.params
        return convert_keypoints_to_dicaugment(
            data,
            params.format,
            rows,
            cols,
            slices,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )


def check_keypoint(kp: Sequence, rows: int, cols: int, slices: int) -> None:
    """
    Checks if keypoint coordinates are less than image shapes or not in correct angle range.
    
    Args:
        data (Sequence): A sequence of keypoint objects
        rows (int): The number of rows in the target image
        cols (int): The number of columns in the target image
        slices (int): The number of slices in the target image
    """
    for name, value, size in zip(["x", "y", "z"], kp[:3], [cols, rows, slices]):
        if not 0 <= value < size:
            raise ValueError(
                "Expected {name} for keypoint {kp} "
                "to be in the range [0.0, {size}], got {value}.".format(
                    kp=kp, name=name, value=value, size=size
                )
            )

    angle = kp[3]
    if not (0 <= angle < 2 * math.pi):
        raise ValueError(
            "Keypoint angle must be in range [0, 2 * PI). Got: {angle}".format(
                angle=angle
            )
        )


def check_keypoints(
    keypoints: Sequence[Sequence], rows: int, cols: int, slices: int
) -> None:
    """Check if keypoints boundaries are less than image shapes"""
    for kp in keypoints:
        check_keypoint(kp, rows, cols, slices)


def filter_keypoints(
    keypoints: Sequence[Sequence],
    rows: int,
    cols: int,
    slices: int,
    remove_invisible: bool,
) -> Sequence[Sequence]:
    """
    Filters out keypoints that are no longer within the bounds of the image
    
    Args:
        keypoints (Sequence): A sequence of keypoint objects
        rows (int): The number of rows in the target image
        cols (int): The number of columns in the target image
        slices (int): The number of slices in the target image
        remove_invisible (bool): Whether to remove keypoints that are no longer within the bounds of the image
    
    Returns:
        keypoints
    
    """
    if not remove_invisible:
        return keypoints

    resulting_keypoints = []
    for kp in keypoints:
        x, y, z = kp[:3]
        if x < 0 or x >= cols:
            continue
        if y < 0 or y >= rows:
            continue
        if z < 0 or z >= slices:
            continue
        resulting_keypoints.append(kp)
    return resulting_keypoints


def convert_keypoint_to_dicaugment(
    keypoint: Sequence,
    source_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> Tuple:
    """
    Converts keypoints to the `dicaugment_3d` format
    
    Args:
        keypoint (Sequence): a sequence representation of a keypoint
        source_format (str): format of keypoints. Should be 'xyz', 'zyx', 'xyza', 'xyzs', 'xyzas', 'xyzsa'.
        rows (int): The number of rows in the target image
        cols (int): The number of columns in the target image
        slices (int): The number of slices in the target image
        check_validity (bool): Whether to check if keypoint coordinates are less than image shapes. Default: False
        angle_in_degrees (bool): Whether the angle of the keypoint is in degrees rather than radians. Default: True

    Returns:
        keypoint converted to the `dicaugment_3d` format
    """
    if source_format not in keypoint_formats:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: {}".format(
                source_format, keypoint_formats
            )
        )

    if source_format == "xyz":
        (x, y, z), tail = keypoint[:3], tuple(keypoint[3:])
        a, s = 0.0, 0.0
    elif source_format == "zyx":
        (z, y, x), tail = keypoint[:3], tuple(keypoint[3:])
        a, s = 0.0, 0.0
    elif source_format == "xyza":
        (x, y, z, a), tail = keypoint[:4], tuple(keypoint[4:])
        s = 0.0
    elif source_format == "xyzs":
        (x, y, z, s), tail = keypoint[:4], tuple(keypoint[4:])
        a = 0.0
    elif source_format == "xyzas":
        (x, y, z, a, s), tail = keypoint[:5], tuple(keypoint[5:])
    elif source_format == "xyzsa":
        (x, y, z, s, a), tail = keypoint[:5], tuple(keypoint[5:])
    else:
        raise ValueError(f"Unsupported source format. Got {source_format}")

    if angle_in_degrees:
        a = math.radians(a)

    keypoint = (x, y, z, angle_to_2pi_range(a), s) + tail
    if check_validity:
        check_keypoint(keypoint, rows, cols, slices)
    return keypoint


def convert_keypoint_from_dicaugment(
    keypoint: Sequence,
    target_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> Tuple:
    """
    Converts keypoints from the `dicaugment_3d` format
    
    Args:
        keypoint (Sequence): a sequence representation of a keypoint
        target_format (str): format of keypoints. Should be 'xyz', 'zyx', 'xyza', 'xyzs', 'xyzas', 'xyzsa'.
        rows (int): The number of rows in the target image
        cols (int): The number of columns in the target image
        slices (int): The number of slices in the target image
        check_validity (bool): Whether to check if keypoint coordinates are less than image shapes. Default: False
        angle_in_degrees (bool): Whether the angle of the keypoint is in degrees rather than radians. Default: True

    Returns:
        keypoint converted from the `dicaugment_3d` format
    """
    if target_format not in keypoint_formats:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: {}".format(
                target_format, keypoint_formats
            )
        )

    (x, y, z, angle, scale), tail = keypoint[:5], tuple(keypoint[5:])
    angle = angle_to_2pi_range(angle)
    if check_validity:
        check_keypoint((x, y, z, angle, scale), rows, cols, slices)
    if angle_in_degrees:
        angle = math.degrees(angle)

    kp: Tuple
    if target_format == "xyz":
        kp = (x, y, z)
    elif target_format == "zyx":
        kp = (z, y, x)
    elif target_format == "xyza":
        kp = (x, y, z, angle)
    elif target_format == "xyzs":
        kp = (x, y, z, scale)
    elif target_format == "xyzas":
        kp = (x, y, z, angle, scale)
    elif target_format == "xyzsa":
        kp = (x, y, z, scale, angle)
    else:
        raise ValueError(f"Invalid target format. Got: {target_format}")

    return kp + tail


def convert_keypoints_to_dicaugment(
    keypoints: Sequence[Sequence],
    source_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> List[Tuple]:
    """
    Converts a sequence of keypoints to the `dicaugment_3d` format
    
    Args:
        keypoint (Sequence): a sequence representation of a keypoint
        source_format (str): format of keypoints. Should be 'xyz', 'zyx', 'xyza', 'xyzs', 'xyzas', 'xyzsa'.
        rows (int): The number of rows in the target image
        cols (int): The number of columns in the target image
        slices (int): The number of slices in the target image
        check_validity (bool): Whether to check if keypoint coordinates are less than image shapes. Default: False
        angle_in_degrees (bool): Whether the angle of the keypoint is in degrees rather than radians. Default: True

    Returns:
        A sequence of keypoints converted to the `dicaugment_3d` format
    """
    return [
        convert_keypoint_to_dicaugment(
            kp, source_format, rows, cols, slices, check_validity, angle_in_degrees
        )
        for kp in keypoints
    ]


def convert_keypoints_from_dicaugment(
    keypoints: Sequence[Sequence],
    target_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> List[Tuple]:
    """
    Converts a sequence of keypoints from the `dicaugment_3d` format
    
    Args:
        keypoint (Sequence): a sequence representation of a keypoint
        target_format (str): format of keypoints. Should be 'xyz', 'zyx', 'xyza', 'xyzs', 'xyzas', 'xyzsa'.
        rows (int): The number of rows in the target image
        cols (int): The number of columns in the target image
        slices (int): The number of slices in the target image
        check_validity (bool): Whether to check if keypoint coordinates are less than image shapes. Default: False
        angle_in_degrees (bool): Whether the angle of the keypoint is in degrees rather than radians. Default: True

    Returns:
        A sequence of keypoints converted from the `dicaugment_3d` format
    """
    return [
        convert_keypoint_from_dicaugment(
            kp, target_format, rows, cols, slices, check_validity, angle_in_degrees
        )
        for kp in keypoints
    ]
