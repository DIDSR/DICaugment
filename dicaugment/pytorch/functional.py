from __future__ import division
import numpy as np
import torch
import torchvision.transforms.functional as F
from typing import Optional, Dict, Sequence

__all__ = [
    "img_to_tensor",
]


def img_to_tensor(
        im: np.ndarray,
        normalize: Optional[Dict[str,Sequence[float]]] = None
    ) -> torch.tensor:
    """Casts a numpy array to a torch.tensor with the option to normalize
    
    Args:
        im (np.ndarray): A numpy array
        normalize (dict, None): Optional keyword argument dictionary for `torchvision.transforms.functional.normalize()`
    
    Returns:
        A torch.tensor
    """
    tensor = torch.from_numpy(im)
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor
