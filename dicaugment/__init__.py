from __future__ import absolute_import

__version__ = "1.0.1"

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *
from .core.bbox_utils import BboxParams
from .core.keypoints_utils import KeypointParams

try:
    from .pytorch.transforms import *
except ImportError:
    # torch is not installed by default, so we import stubs.
    # Run `pip install -U albumentations[torch] if you need augmentations from torch.`
    from .pytorch.stubs import *  # type: ignore

try:
    from .tensorflow.transforms import *
except ImportError:
    # tensorflow is not installed by default, so we import stubs.
    # Run `pip install -U albumentations[tensorflow] if you need augmentations from tensorflow.`
    from .tensorflow.stubs import *  # type: ignore
