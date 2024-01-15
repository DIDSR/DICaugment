import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np

from ...core.transforms_interface import DualTransform, KeypointType
from .functional import cutout

__all__ = ["CoarseDropout"]


class CoarseDropout(DualTransform):
    """CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
            If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
            If float, it is calculated as a fraction of the image width.
        max_depth (int, float): Maximum depth of the hole.
            If float, it is calculated as a fraction of the image depth.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_width` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.
        min_depth (int, float): Minimum depth of the hole. If `None`, `min_depth` is
            set to `max_depth`. Default: `None`.
            If float, it is calculated as a fraction of the image depth.
        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.
        always_apply (bool): whether to always apply the transformation. Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        max_depth: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        min_depth: Optional[int] = None,
        fill_value: int = 0,
        mask_fill_value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(CoarseDropout, self).__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.max_depth = max_depth
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.min_depth = min_depth if min_depth is not None else max_depth
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError(
                "Invalid combination of min_holes and max_holes. Got: {}".format(
                    [min_holes, max_holes]
                )
            )

        self.check_range(self.max_height)
        self.check_range(self.min_height)
        self.check_range(self.max_width)
        self.check_range(self.min_width)
        self.check_range(self.max_depth)
        self.check_range(self.min_depth)

        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format(
                    [min_height, max_height]
                )
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError(
                "Invalid combination of min_width and max_width. Got: {}".format(
                    [min_width, max_width]
                )
            )
        if not 0 < self.min_depth <= self.max_depth:
            raise ValueError(
                "Invalid combination of min_depth and max_depth. Got: {}".format(
                    [min_depth, max_depth]
                )
            )

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(
                    dimension
                )
            )

    def apply(
        self,
        img: np.ndarray,
        fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int, int, int]] = (),
        **params
    ) -> np.ndarray:
        """Applies the transformation to the image"""
        return cutout(img, holes, fill_value)

    def apply_to_mask(
        self,
        img: np.ndarray,
        mask_fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int, int, int]] = (),
        **params
    ) -> np.ndarray:
        """Applies the transformation to a mask"""
        if mask_fill_value is None:
            return img
        return cutout(img, holes, mask_fill_value)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns additional parameters needed for the `apply` methods that depend on a target
        (e.g. `apply_to_bboxes` method expects image size)
        """
        img = params["image"]
        height, width, depth = img.shape[:3]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            if all(
                [
                    isinstance(self.min_height, int),
                    isinstance(self.min_width, int),
                    isinstance(self.max_depth, int),
                    isinstance(self.max_height, int),
                    isinstance(self.max_width, int),
                    isinstance(self.max_depth, int),
                ]
            ):
                hole_height = random.randint(self.min_height, self.max_height)
                hole_width = random.randint(self.min_width, self.max_width)
                hole_depth = random.randint(self.min_depth, self.max_depth)
            elif all(
                [
                    isinstance(self.min_height, float),
                    isinstance(self.min_width, float),
                    isinstance(self.min_depth, float),
                    isinstance(self.max_height, float),
                    isinstance(self.max_width, float),
                    isinstance(self.max_depth, float),
                ]
            ):
                hole_height = int(
                    height * random.uniform(self.min_height, self.max_height)
                )
                hole_width = int(width * random.uniform(self.min_width, self.max_width))
                hole_depth = int(width * random.uniform(self.min_depth, self.max_depth))
            else:
                raise ValueError(
                    "Min width, max width, \
                    min height and max height \
                    should all either be ints or floats. \
                    Got: {} respectively".format(
                        [
                            type(self.min_width),
                            type(self.max_width),
                            type(self.min_height),
                            type(self.max_height),
                            type(self.min_depth),
                            type(self.max_depth),
                        ]
                    )
                )

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            z1 = random.randint(0, depth - hole_depth)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            z2 = z1 + hole_depth
            holes.append((x1, y1, z1, x2, y2, z2))

        return {"holes": holes}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def _keypoint_in_hole(
        self, keypoint: KeypointType, hole: Tuple[int, int, int, int]
    ) -> bool:
        x1, y1, z1, x2, y2, z2 = hole
        x, y, z = keypoint[:3]
        return x1 <= x < x2 and y1 <= y < y2 and z1 <= z <= z2

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params
    ) -> List[KeypointType]:
        """Applies the transformation to a sequence of keypoints"""
        result = set(keypoints)
        for hole in holes:
            for kp in keypoints:
                if self._keypoint_in_hole(kp, hole):
                    result.discard(kp)
        return list(result)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Returns initialization argument names. (e.g. Transform(arg1 = 1, arg2 = 2) -> ('arg1', 'arg2'))"""
        return (
            "max_holes",
            "max_height",
            "max_width",
            "max_depth",
            "min_holes",
            "min_height",
            "min_width",
            "min_depth",
            "fill_value",
            "mask_fill_value",
        )
