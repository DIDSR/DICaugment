from __future__ import division
import numpy as np
import torch
import torchvision.transforms.functional as F

__all__ = [
    "img_to_tensor",
    "mask_to_tensor",
]

def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(im)
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def mask_to_tensor(mask, num_classes, sigmoid):
    if num_classes > 1:
        if not sigmoid:
            # softmax
            long_mask = np.zeros((mask.shape[:3]), dtype=np.int64)
            if len(mask.shape) == 4:
                for c in range(mask.shape[3]):
                    long_mask[mask[..., c] > 0] = c
            else:
                long_mask[mask > 127] = 1
                long_mask[mask == 0] = 0
            mask = long_mask
        else:
            mask = np.moveaxis(mask / (255.0 if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    else:
        mask = np.expand_dims(mask / (255.0 if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    return torch.from_numpy(mask)
