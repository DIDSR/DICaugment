from typing import Iterable, List, Tuple, Union

import numpy as np

from dicaugment.augmentations.utils import preserve_shape

__all__ = [
    "cutout",
]

def cutout(
    img: np.ndarray,
    holes: Iterable[Tuple[int, int, int, int, int, int]],
    fill_value: Union[int, float] = 0,
) -> np.ndarray:
    """
    Puts holes in image.

    Args:
        img (np.ndarray): an image
        holes (Sequence of Tuples): A sequence of indexing ranges for each dimension to create a hole in
        fill_value (int,float): The value to fill the holes with. Default: 0
    """
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()
    for x1, y1, z1, x2, y2, z2 in holes:
        img[y1:y2, x1:x2, z1:z2] = fill_value
    return img
