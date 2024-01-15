import pytest

import numpy as np
import torch
from PIL import Image

import dicaugment as A


def test_torch_to_tensor_augmentations(image, mask):
    aug = A.ToPytorch()
    data = aug(image=image, mask=mask, force_apply=True)
    height, width, num_channels = image.shape
    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (
        1,
        num_channels,
        height,
        width,
    )
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == mask.shape
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


def test_torch_to_tensor_v2_augmentations_with_transpose_2d_mask(image, mask):
    aug = A.ToPytorch(transpose_mask=True)
    data = aug(image=image, mask=mask, force_apply=True)
    image_height, image_width, image_depth = image.shape
    mask_height, mask_width, mask_depth = mask.shape
    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (
        1,
        image_depth,
        image_height,
        image_width,
    )
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == (
        mask_height,
        mask_width,
        mask_depth,
    )
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


def test_torch_to_tensor_v2_augmentations_with_transpose_3d_mask(image):
    aug = A.ToPytorch(transpose_mask=True)
    mask = np.random.randint(low=0, high=256, size=(50, 50, 50, 4), dtype=np.uint8)
    data = aug(image=image, mask=mask, force_apply=True)
    image_height, image_width, image_depth = image.shape
    mask_height, mask_width, mask_depth, mask_num_channels = mask.shape
    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == (
        1,
        image_depth,
        image_height,
        image_width,
    )
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == (
        mask_num_channels,
        mask_depth,
        mask_height,
        mask_width,
    )
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


def test_additional_targets_for_ToPytorch():
    aug = A.Compose(
        [A.ToPytorch(transpose_mask=False)],
        additional_targets={"image2": "image", "mask2": "mask"},
    )
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(50, 50, 50), dtype=np.uint8)
        image2 = image1.copy()
        mask1 = np.random.randint(low=0, high=256, size=(50, 50, 50, 4), dtype=np.uint8)
        mask2 = mask1.copy()
        res = aug(image=image1, image2=image2, mask=mask1, mask2=mask2)

        image1_height, image1_width, image1_depth = image1.shape
        image2_height, image2_width, image2_depth = image2.shape
        assert isinstance(res["image"], torch.Tensor) and res["image"].shape == (
            1,
            image1_depth,
            image1_height,
            image1_width,
        )
        assert isinstance(res["image2"], torch.Tensor) and res["image2"].shape == (
            1,
            image2_depth,
            image2_height,
            image2_width,
        )
        assert (
            isinstance(res["mask"], torch.Tensor) and res["mask"].shape == mask1.shape
        )
        assert (
            isinstance(res["mask2"], torch.Tensor) and res["mask2"].shape == mask2.shape
        )
        assert np.array_equal(res["image"], res["image2"])
        assert np.array_equal(res["mask"], res["mask2"])


def test_torch_to_tensor_v2_on_gray_scale_images():
    aug = A.ToPytorch()
    grayscale_image = np.random.randint(
        low=0, high=256, size=(100, 100, 50), dtype=np.uint8
    )
    data = aug(image=grayscale_image)
    assert isinstance(data["image"], torch.Tensor)
    assert len(data["image"].shape) == 4
    # assert data["image"].shape[1:] == grayscale_image.shape
    assert data["image"].dtype == torch.uint8


def test_with_replaycompose():
    aug = A.ReplayCompose([A.ToPytorch()])
    kwargs = {
        "image": np.random.randint(
            low=0, high=256, size=(100, 100, 100), dtype=np.uint8
        ),
        "mask": np.random.randint(
            low=0, high=256, size=(100, 100, 100), dtype=np.uint8
        ),
    }
    res = aug(**kwargs)
    res2 = A.ReplayCompose.replay(res["replay"], **kwargs)
    assert np.array_equal(res["image"], res2["image"])
    assert np.array_equal(res["mask"], res2["mask"])
    assert res["image"].dtype == torch.uint8
    assert res["mask"].dtype == torch.uint8
    assert res2["image"].dtype == torch.uint8
    assert res2["mask"].dtype == torch.uint8


def test_post_data_check():
    img = np.empty([100, 100, 100], dtype=np.uint8)
    bboxes = [
        [0, 0, 0, 90, 90, 90, 0],
    ]
    keypoints = [
        [90, 90, 90],
        [50, 50, 50],
    ]

    transform = A.Compose(
        [
            A.Resize(50, 50, 50),
            A.Normalize(),
            A.ToPytorch(),
        ],
        keypoint_params=A.KeypointParams("xyz"),
        bbox_params=A.BboxParams("pascal_voc_3d"),
    )

    res = transform(image=img, keypoints=keypoints, bboxes=bboxes)
    assert res["keypoints"] == [(45, 45, 45), (25, 25, 25)]
    assert res["bboxes"] == [(0, 0, 0, 45, 45, 45, 0)]
    assert len(res["keypoints"]) != 0 and len(res["bboxes"]) != 0
