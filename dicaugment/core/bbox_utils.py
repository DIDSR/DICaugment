from __future__ import division

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np

from .transforms_interface import BoxInternalType, BoxType
from .utils import DataProcessor, Params

__all__ = [
    "normalize_bbox",
    "denormalize_bbox",
    "normalize_bboxes",
    "denormalize_bboxes",
    "calculate_bbox_area_volume",
    "filter_bboxes_by_visibility",
    "convert_bbox_to_dicaugment",
    "convert_bbox_from_dicaugment",
    "convert_bboxes_to_dicaugment",
    "convert_bboxes_from_dicaugment",
    "check_bbox",
    "check_bboxes",
    "filter_bboxes",
    "union_of_bboxes",
    "BboxProcessor",
    "BboxParams",
    "BB_COCO_3D",
    "BB_PASCAL_VOC_3D",
    "BB_dicaugment_3D",
    "BB_YOLO_3D",
]

TBox = TypeVar("TBox", BoxType, BoxInternalType)

BB_COCO_3D = "coco_3d"
BB_PASCAL_VOC_3D = "pascal_voc_3d"
BB_dicaugment_3D = "dicaugment_3d"
BB_YOLO_3D = "yolo_3d"


class BboxParams(Params):
    """
    Parameters of bounding boxes

    Args:
        format (str): format of bounding boxes. Should be 'coco_3d', 'pascal_voc_3d', 'dicaugment_3d' or 'yolo_3d'.

            The `coco_3d` format
                `[x_min, y_min, z_min, width, height, depth]`, e.g. [97, 12, 5, 150, 200, 10].
            The `pascal_voc_3d` format
                `[x_min, y_min, z_min, x_max, y_max, z_min]`, e.g. [97, 12, 5, 247, 212, 10].
            The `dicaugment_3d` format
                is like `pascal_voc_3d`, but normalized,
                in other words: `[x_min, y_min, z_min, x_max, y_max, z_min]`, e.g. [0.2, 0.3, 0.5, 0.4, 0.5, 0.8].
            The `yolo_3d` format
                `[x, y, z, width, height, depth]`, e.g. [0.3, 0.4, 0.5, 0.1, 0.2, 0.3];
                `x`, `y`, `z` - normalized bbox center; `width`, `height`, `depth` - normalized bbox width, height, and depth

            You may also pass a predefined string such as albumentation3D.BB_COCO_3D or
        label_fields (list): list of fields that are joined with boxes, e.g labels.
            Should be same type as boxes.
        min_planar_area (float): minimum area of a bounding box for a single slice. All bounding boxes whose
            visible area in pixels is less than this value will be removed. Default: 0.0.
        min_volume (float): minimum volume of a bounding box. All bounding boxes whose
            visible volume in pixels is less than this value will be removed. Default: 0.0.
            This assumes that pixel spacing of the Height and Width dimensions are equal to the slice spacing of the Depth dimension
        min_area_visibility (float): minimum fraction of planar area for a bounding box
            to remain this box in list. Default: 0.0.
        min_volume_visibility (float): minimum fraction of volume for a bounding box
            to remain this box in list. Default: 0.0.
        min_width (float): Minimum width of a bounding box. All bounding boxes whose width is
            less than this value will be removed. Default: 0.0.
        min_height (float): Minimum height of a bounding box. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.
        min_depth (float): Minimum depth of a bounding box. All bounding boxes whose depth is
            less than this value will be removed. Default: 0.0.
        check_each_transform (bool): if `True`, then bboxes will be checked after each dual transform.
            Default: `True`
    """

    def __init__(
        self,
        format: str,
        label_fields: Optional[Sequence[str]] = None,
        min_planar_area: float = 0.0,
        min_volume: float = 0.0,
        min_area_visibility: float = 0.0,
        min_volume_visibility: float = 0.0,
        min_width: float = 0.0,
        min_height: float = 0.0,
        min_depth: float = 0.0,
        check_each_transform: bool = True,
    ):
        super(BboxParams, self).__init__(format, label_fields)
        self.min_planar_area = min_planar_area
        self.min_volume = min_volume
        self.min_area_visibility = min_area_visibility
        self.min_volume_visibility = min_volume_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.min_depth = min_depth
        self.check_each_transform = check_each_transform

    def _to_dict(self) -> Dict[str, Any]:
        data = super(BboxParams, self)._to_dict()
        data.update(
            {
                "min_planar_area": self.min_planar_area,
                "min_volume": self.min_volume,
                "min_area_visibility": self.min_area_visibility,
                "min_volume_visibility": self.min_volume_visibility,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "min_depth": self.min_depth,
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
        return "BboxParams"


class BboxProcessor(DataProcessor):
    """
    Processor Class for Bounding Boxes

    Args:
        params (BboxParams): An instance of BboxParams
        additional_targets (dict): keys - new target name, values - old target name. ex: {'bboxes2': 'bboxes'}
    """

    def __init__(
        self, params: BboxParams, additional_targets: Optional[Dict[str, str]] = None
    ):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        """Returns the default data name for class (e.g. 'image')"""
        return "bboxes"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        """Raises a ValueError if input data is not in the expected format
        (e.g. `label_fields` does not match up with values with params dict)"""
        for data_name in self.data_fields:
            data_exists = data_name in data and len(data[data_name])
            if data_exists and len(data[data_name][0]) < 7:
                if self.params.label_fields is None:
                    raise ValueError(
                        "Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                        "because bboxes must have labels"
                    )
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same names as params in dict"
                )

    def filter(self, data: Sequence, rows: int, cols: int, slices: int) -> List:
        """
        Wrapper method that invokes `filter_bboxes`. 
        Remove bounding boxes that either lie outside of the visible planar area or 
        volume by more then min_area_visibility or min_volume_visibility, respectively, 
        as well as any bounding boxes or whose planar area/volumne in pixels is under
        the threshold set by `min_area` and `min_volume`. Also it crops boxes to final image size.
    
        Args:
            data (Sequence): A sequence of bbox objects
            rows (int): The number of rows in the target image
            cols (int): The number of columns in the target image
            slices (int): The number of slices in the target image
        
        Returns:
            bboxes
        """
        self.params: BboxParams
        return filter_bboxes(
            data,
            rows,
            cols,
            slices,
            min_planar_area=self.params.min_planar_area,
            min_volume=self.params.min_volume,
            min_area_visibility=self.params.min_area_visibility,
            min_volume_visibility=self.params.min_volume_visibility,
            min_width=self.params.min_width,
            min_height=self.params.min_height,
            min_depth=self.params.min_depth,
        )

    def check(self, data: Sequence, rows: int, cols: int, slices: int) -> None:
        """Wrapper method that invokes `check_keypoints`.
        Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums
        """
        check_bboxes(data)

    def convert_from_dicaugment(
        self, data: Sequence, rows: int, cols: int, slices: int
    ) -> List[BoxType]:
        """
        Convert a bounding box from the format used by dicaugment to a format, specified in `params.format`.

        Args:
            data (Sequence): A sequence of bounding boxes.
            rows: Image height.
            cols: Image width.
            slices: Image depth
        Returns:
            Sequence: A sequence of bounding boxes
        """
        return convert_bboxes_from_dicaugment(
            data, self.params.format, rows, cols, slices, check_validity=True
        )

    def convert_to_dicaugment(
        self, data: Sequence[BoxType], rows: int, cols: int, slices: int
    ) -> List[BoxType]:
        """
        Convert a bounding box from a format specified in `params.format` to the format used by dicaugment:
        normalized coordinates of closest top-left and furthest bottom-right corners of the bounding box in a form of
        `(x_min, y_min, z_min, x_max, y_max, z_max)` e.g. `(0.15, 0.27, 0.12, 0.67, 0.5, 0.48)`.

        Args:
            data (Sequence): A sequence of bounding boxes.
            rows: Image height.
            cols: Image width.
            slices: Image depth
        Returns:
            Sequence: A sequence of bounding boxes
        """
        return convert_bboxes_to_dicaugment(
            data, self.params.format, rows, cols, slices, check_validity=True
        )


def normalize_bbox(bbox: TBox, rows: int, cols: int, slices: int) -> TBox:
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width, y-coordinates
    by image height, and z-coordinates by image depth

    Args:
        bbox: Denormalized bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image height.
        cols: Image width.
        slices : Image depth.

    Returns:
        Normalized bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    Raises:
        ValueError: If rows, cols, or slices is less or equal zero

    """

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")
    if slices <= 0:
        raise ValueError("Argument slices must be positive integer")

    tail: Tuple[Any, ...]
    (x_min, y_min, z_min, x_max, y_max, z_max), tail = bbox[:6], tuple(bbox[6:])

    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows
    z_min, z_max = z_min / slices, z_max / slices

    return cast(BoxType, (x_min, y_min, z_min, x_max, y_max, z_max) + tail)  # type: ignore


def denormalize_bbox(bbox: TBox, rows: int, cols: int, slices: int) -> TBox:
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width, y-coordinates
    by image height, and z-coordinates by image depth.
    This is an inverse operation for :func:`~dicaugment.augmentations.bbox.normalize_bbox`.

    Args:
        bbox: Normalized bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image height.
        cols: Image width.
        slices : Image depth.

    Returns:
        Denormalized bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    Raises:
        ValueError: If rows, cols, or slices is less or equal zero

    """
    tail: Tuple[Any, ...]
    (x_min, y_min, z_min, x_max, y_max, z_max), tail = bbox[:6], tuple(bbox[6:])

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")
    if slices <= 0:
        raise ValueError("Argument slices must be positive integer")

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows
    z_min, z_max = z_min * slices, z_max * slices

    return cast(BoxType, (x_min, y_min, z_min, x_max, y_max, z_max) + tail)  # type: ignore


def normalize_bboxes(
    bboxes: Sequence[BoxType], rows: int, cols: int, slices: int
) -> List[BoxType]:
    """Normalize a list of bounding boxes.

    Args:
        bboxes: Denormalized bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.
        rows: Image height.
        cols: Image width.
        slices: Image depth

    Returns:
        Normalized bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.

    """
    return [normalize_bbox(bbox, rows, cols, slices) for bbox in bboxes]


def denormalize_bboxes(
    bboxes: Sequence[BoxType], rows: int, cols: int, slices: int
) -> List[BoxType]:
    """Denormalize a list of bounding boxes.

    Args:
        bboxes: Normalized bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.
        rows: Image height.
        cols: Image width.
        slices: Image depth.

    Returns:
        List: Denormalized bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.

    """
    return [denormalize_bbox(bbox, rows, cols, slices) for bbox in bboxes]


def calculate_bbox_area_volume(
    bbox: BoxType, rows: int, cols: int, slices: int
) -> Tuple[float, float]:
    """Calculate the planar area and volume of a bounding box in (fractional) pixels for a slice.

    Args:
        bbox: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image height.
        cols: Image width.
        slices: Image depth

    Return:
        Planar area and volume in (fractional) pixels of the (denormalized) bounding box.

    """
    bbox = denormalize_bbox(bbox, rows, cols, slices)
    (x_min, y_min, z_min, x_max, y_max, z_max) = bbox[:6]
    area = (x_max - x_min) * (y_max - y_min)
    volume = area * (z_max - z_min)
    return area, volume


def filter_bboxes_by_visibility(
    original_shape: Sequence[int],
    bboxes: Sequence[BoxType],
    transformed_shape: Sequence[int],
    transformed_bboxes: Sequence[BoxType],
    area_threshold: float = 0.0,
    volume_threshold: float = 0.0,
    min_area: float = 0.0,
    min_volume: float = 0.0,
) -> List[BoxType]:
    """Filter bounding boxes and return only those boxes whose visibility after transformation is above
    the threshold and minimal area of bounding box in pixels is more then min_area.

    Args:
        original_shape: Original image shape `(height, width, depth, ...)`.
        bboxes: Original bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.
        transformed_shape: Transformed image shape `(height, width, depth)`.
        transformed_bboxes: Transformed bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.
        area_threshold: planar area visibility threshold. Should be a value in the range [0.0, 1.0].
        volume_threshold: volume visibility threshold. Should be a value in the range [0.0, 1.0].
        min_area: Minimal area threshold.
        min_volumne: Minimal volume theshold

    Returns:
        Filtered bounding boxes `[(x_min, y_min, z_min, x_max, y_max, z_max)]`.

    """
    img_height, img_width, img_depth = original_shape[:3]
    (
        transformed_img_height,
        transformed_img_width,
        transformed_img_depth,
    ) = transformed_shape[:3]

    visible_bboxes = []
    for bbox, transformed_bbox in zip(bboxes, transformed_bboxes):
        if not all(0.0 <= value <= 1.0 for value in transformed_bbox[:6]):
            continue
        bbox_area, bbox_volume = calculate_bbox_area_volume(
            bbox, img_height, img_width, img_depth
        )
        transformed_bbox_area, transformed_bbox_volume = calculate_bbox_area_volume(
            transformed_bbox,
            transformed_img_height,
            transformed_img_width,
            transformed_img_depth,
        )

        if transformed_bbox_area < min_area:
            continue
        if transformed_bbox_volume < min_volume:
            continue
        area_visibility = transformed_bbox_area / bbox_area
        volume_visibility = transformed_bbox_volume / bbox_volume

        if area_visibility >= area_threshold and volume_visibility >= volume_threshold:
            visible_bboxes.append(transformed_bbox)
    return visible_bboxes


def convert_bbox_to_dicaugment(
    bbox: BoxType,
    source_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
) -> BoxType:
    """Convert a bounding box from a format specified in `source_format` to the format used by dicaugment:
    normalized coordinates of closest top-left and furthest bottom-right corners of the bounding box in a form of
    `(x_min, y_min, z_min, x_max, y_max, z_max)` e.g. `(0.15, 0.27, 0.12, 0.67, 0.5, 0.48)`.

    Args:
        bbox: A bounding box tuple.
        source_format: format of the bounding box. Should be 'coco_3d', 'pascal_voc_3d', or 'yolo_3d'.
        check_validity: Check if all boxes are valid boxes.
        rows: Image height.
        cols: Image width.
        slices: Image depth
    Returns:
        tuple: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
    Note:
        The `coco_3d` format of a bounding box looks like `(x_min, y_min, z_min, width, height, depth)`, e.g. ([97, 12, 5, 150, 200, 10).
        The `pascal_voc_3d` format of a bounding box looks like `(x_min, y_min, z_min, x_max, y_max, z_min)`, e.g. (97, 12, 5, 247, 212, 15).
        The `yolo_3d` format of a bounding box looks like `(x, y, z, width, height, depth)`, e.g. (0.3, 0.1, 0.5, 0.05, 0.07, 0.23) where `x`, `y`, and `z` are coordinates of the center of the box, all values normalized to 1 by image height, width, and depth.

    Raises:
        ValueError: if `target_format` is not equal to `coco_3d` or `pascal_voc_3d`, or `yolo_3d`.
        ValueError: If in YOLO format all labels not in range (0, 1).

    """
    if source_format not in {"coco_3d", "pascal_voc_3d", "yolo_3d"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco_3d', 'pascal_voc_3d' and 'yolo_3d'"
        )

    if source_format == "coco_3d":
        (x_min, y_min, z_min, width, height, depth), tail = bbox[:6], bbox[6:]
        x_max = x_min + width
        y_max = y_min + height
        z_max = z_min + depth
    elif source_format == "yolo_3d":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        _bbox = np.array(bbox[:6])
        if check_validity and np.any((_bbox <= 0) | (_bbox > 1)):
            raise ValueError(
                "In YOLO format all coordinates must be float and in range (0, 1]"
            )

        (x, y, z, w, h, d), tail = bbox[:6], bbox[6:]

        w_half, h_half, d_half = w / 2, h / 2, d / 2
        x_min = x - w_half
        y_min = y - h_half
        z_min = z - d_half
        x_max = x_min + w
        y_max = y_min + h
        z_max = z_min + d
    else:
        (x_min, y_min, z_min, x_max, y_max, z_max), tail = bbox[:6], tuple(bbox[6:])

    bbox = (x_min, y_min, z_min, x_max, y_max, z_max) + tuple(tail)  # type: ignore

    if source_format != "yolo_3d":
        bbox = normalize_bbox(bbox, rows, cols, slices)
    if check_validity:
        check_bbox(bbox)
    return bbox


def convert_bbox_from_dicaugment(
    bbox: BoxType,
    target_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
) -> BoxType:
    """Convert a bounding box from the format used by dicaugment to a format, specified in `target_format`.

    Args:
        bbox: An dicaugment bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        target_format: required format of the output bounding box. Should be 'coco_3d', 'pascal_voc_3d' or 'yolo_3d'.
        rows: Image height.
        cols: Image width.
        slices: Image depth.
        check_validity: Check if all boxes are valid boxes.
    Returns:
        tuple: A bounding box.
    Note:
        The `coco_3d` format of a bounding box looks like `(x_min, y_min, z_min, width, height, depth)`, e.g. ([97, 12, 5, 150, 200, 10).
        The `pascal_voc_3d` format of a bounding box looks like `(x_min, y_min, z_min, x_max, y_max, z_min)`, e.g. (97, 12, 5, 247, 212, 15).
        The `yolo_3d` format of a bounding box looks like `(x, y, z, width, height, depth)`, e.g. (0.3, 0.1, 0.5, 0.05, 0.07, 0.23) where `x`, `y`, and `z` are coordinates of the center of the box, all values normalized to 1 by image height, width, and depth.
    Raises:
        ValueError: if `target_format` is not equal to `coco_3d`, `pascal_voc_3d` or `yolo_3d`.

    """
    if target_format not in {"coco_3d", "pascal_voc_3d", "yolo_3d"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are: 'coco_3d', 'pascal_voc_3d' and 'yolo_3d'"
        )
    if check_validity:
        check_bbox(bbox)

    if target_format != "yolo_3d":
        bbox = denormalize_bbox(bbox, rows, cols, slices)
    if target_format == "coco_3d":
        (x_min, y_min, z_min, x_max, y_max, z_max), tail = bbox[:6], tuple(bbox[6:])
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min
        bbox = cast(BoxType, (x_min, y_min, z_min, width, height, depth) + tail)
    elif target_format == "yolo_3d":
        (x_min, y_min, z_min, x_max, y_max, z_max), tail = bbox[:6], tuple(bbox[6:])
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        z = (z_min + z_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        d = z_max - z_min
        bbox = cast(BoxType, (x, y, z, w, h, d) + tail)
    return bbox


def convert_bboxes_to_dicaugment(
    bboxes: Sequence[BoxType],
    source_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity=False,
) -> List[BoxType]:
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by dicaugment"""
    return [
        convert_bbox_to_dicaugment(
            bbox, source_format, rows, cols, slices, check_validity
        )
        for bbox in bboxes
    ]


def convert_bboxes_from_dicaugment(
    bboxes: Sequence[BoxType],
    target_format: str,
    rows: int,
    cols: int,
    slices: int,
    check_validity: bool = False,
) -> List[BoxType]:
    """Convert a list of bounding boxes from the format used by dicaugment to a format, specified
    in `target_format`.

    Args:
        bboxes: List of albumentation bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        target_format: required format of the output bounding box. Should be 'coco_3d', 'pascal_voc_3d' or 'yolo_3d'.
        rows: Image height.
        cols: Image width.
        slices: Image depth
        check_validity: Check if all boxes are valid boxes.

    Returns:
        List of bounding boxes.

    """
    return [
        convert_bbox_from_dicaugment(
            bbox, target_format, rows, cols, slices, check_validity
        )
        for bbox in bboxes
    ]


def check_bbox(bbox: BoxType) -> None:
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(
        ["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"], bbox[:6]
    ):
        if (
            not 0 <= value <= 1
            and not np.isclose(value, 0)
            and not np.isclose(value, 1)
        ):
            raise ValueError(
                f"Expected {name} for bbox {bbox} to be in the range [0.0, 1.0], got {value}."
            )
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[:6]
    if x_max <= x_min:
        raise ValueError(f"x_max is less than or equal to x_min for bbox {bbox}.")
    if y_max <= y_min:
        raise ValueError(f"y_max is less than or equal to y_min for bbox {bbox}.")
    if z_max <= z_min:
        raise ValueError(f"z_max is less than or equal to z_min for bbox {bbox}.")


def check_bboxes(bboxes: Sequence[BoxType]) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)


def filter_bboxes(
    bboxes: Sequence[BoxType],
    rows: int,
    cols: int,
    slices: int,
    min_area_visibility: float = 0.0,
    min_volume_visibility: float = 0.0,
    min_planar_area: float = 0.0,
    min_volume: float = 0.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
    min_depth: float = 0.0,
) -> List[BoxType]:
    """Remove bounding boxes that either lie outside of the visible planar area or volume by more then min_area_visibility
    or min_volume_visibility, respectively, as well as any bounding boxes or whose planar area/volumne in pixels is under
    the threshold set by `min_area` and `min_volume`. Also it crops boxes to final image size.

    Args:
        bboxes: List of dicaugment bounding boxes `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        rows: Image height.
        cols: Image width.
        slices: Image depth.
        min_planar_area: Minimum planar area of a bounding box. All bounding boxes whose visible planar area in pixels.
            is less than this value will be removed. Default: 0.0.
        min_volume: Minimum volume of a bounding box. All bounding boxes whose visible volume in pixels.
            is less than this value will be removed. Default: 0.0.
        min_area_visibility: Minimum fraction of planar area for a bounding box to remain this box in list. Default: 0.0.
        min_volume_visibility: Minimum fraction of volume for a bounding box to remain this box in list. Default: 0.0.
        min_width: Minimum width of a bounding box in pixels. All bounding boxes whose width is
            less than this value will be removed. Default: 0.0.
        min_height: Minimum height of a bounding box in pixels. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.
        min_depth: Minimum depth of a bounding box in pixels. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.

    Returns:
        List of bounding boxes.

    """
    resulting_boxes: List[BoxType] = []
    for bbox in bboxes:
        # Calculate areas of bounding box before and after clipping.
        transformed_box_area, transformed_box_volume = calculate_bbox_area_volume(
            bbox, rows, cols, slices
        )
        bbox, tail = cast(BoxType, tuple(np.clip(bbox[:6], 0, 1.0))), tuple(bbox[6:])
        clipped_box_area, clipped_box_volume = calculate_bbox_area_volume(
            bbox, rows, cols, slices
        )

        # Calculate width and height of the clipped bounding box.
        x_min, y_min, z_min, x_max, y_max, z_max = denormalize_bbox(
            bbox, rows, cols, slices
        )[:6]
        clipped_width, clipped_height, clipped_depth = (
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )

        if (
            clipped_box_area
            != 0  # to ensure transformed_box_area!=0 and to handle min_area=0 or min_visibility=0
            and clipped_box_area >= min_planar_area
            and clipped_box_volume >= min_volume
            and clipped_box_area / transformed_box_area >= min_area_visibility
            and clipped_box_volume / transformed_box_volume >= min_volume_visibility
            and clipped_width >= min_width
            and clipped_height >= min_height
            and clipped_depth >= min_depth
        ):
            resulting_boxes.append(cast(BoxType, bbox + tail))
    return resulting_boxes


def union_of_bboxes(
    height: int,
    width: int,
    depth: int,
    bboxes: Sequence[BoxType],
    erosion_rate: float = 0.0,
) -> BoxType:
    """Calculate union of bounding boxes.

    Args:

        height: Height of image or space.
        width: Width of image or space.
        depth: Depth of image or space.
        bboxes: List of dicaugment bounding boxes `(x_min, y_min, z_min, x_max, y_max, z_max)`.
        erosion_rate: How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, z_min, x_max, y_max, z_max)`.

    """
    x1, y1, z1 = width, height, depth
    x2, y2, z2 = 0, 0, 0
    for bbox in bboxes:
        (x_min, y_min, z_min, x_max, y_max, z_max) = bbox[:6]
        w, h, d = x_max - x_min, y_max - y_min, z_max - z_min
        lim_x1, lim_y1, lim_z1 = (
            x_min + erosion_rate * w,
            y_min + erosion_rate * h,
            z_min + erosion_rate * d,
        )
        lim_x2, lim_y2, lim_z2 = (
            x_max - erosion_rate * w,
            y_max - erosion_rate * h,
            z_max - erosion_rate * d,
        )
        x1, y1, z1 = np.min([x1, lim_x1]), np.min([y1, lim_y1]), np.min([z1, lim_z1])
        x2, y2, z2 = np.max([x2, lim_x2]), np.max([y2, lim_y2]), np.max([z2, lim_z2])
    return x1, y1, z1, x2, y2, z2
