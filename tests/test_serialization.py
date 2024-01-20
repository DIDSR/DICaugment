import json
import os
import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import dicaugment as A
import dicaugment.augmentations.functional as F
import dicaugment.augmentations.geometric.functional as FGeometric
from dicaugment.core.serialization import SERIALIZABLE_REGISTRY, shorten_class_name
from dicaugment.core.transforms_interface import ImageOnlyTransform

from .conftest import skipif_no_torch
from .utils import (
    OpenMock,
    check_all_augs_exists,
    get_image_only_transforms,
    get_transforms,
    set_seed,
)

TEST_SEEDS = (0, 1, 42, 111, 9999)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.LongestMaxSize,
            A.SmallestMaxSize,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization(
    augmentation_cls, params, p, seed, image, mask, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


AUGMENTATION_CLS_PARAMS = [
    [A.RandomBrightnessContrast, {"brightness_limit": 0.5, "contrast_limit": 0.8}],
    [A.Blur, {"blur_limit": 3}],
    [A.MedianBlur, {"blur_limit": 3}],
    [A.GaussianBlur, {"blur_limit": 3}],
    [A.GaussNoise, {"var_limit": (20, 90), "mean": 10, "per_channel": False}],
    [A.RandomGamma, {"gamma_limit": (10, 90)}],
    [A.CoarseDropout, {"max_holes": 4, "max_height": 4, "max_width": 4}],
    [
        A.PadIfNeeded,
        {
            "min_height": 128,
            "min_width": 128,
            "min_depth": 128,
            "border_mode": "constant",
            "value": 10,
        },
    ],
    [
        A.Rotate,
        {
            "limit": 120,
            "interpolation": 2,
            "border_mode": "constant",
            "value": 10,
        },
    ],
    [
        A.ShiftScaleRotate,
        {
            "shift_limit": 0.2,
            "scale_limit": 0.2,
            "rotate_limit": 70,
            "interpolation": 2,
            "border_mode": "constant",
            "value": 10,
        },
    ],
    [
        A.ShiftScaleRotate,
        {
            "shift_limit_x": 0.3,
            "shift_limit_y": 0.4,
            "shift_limit_z": 0.1,
            "scale_limit": 0.2,
            "rotate_limit": 70,
            "interpolation": 2,
            "border_mode": "constant",
            "value": 10,
        },
    ],
    [A.CenterCrop, {"height": 10, "width": 10, "depth": 10}],
    [A.RandomCrop, {"height": 10, "width": 10, "depth": 10}],
    [
        A.RandomSizedCrop,
        {"min_max_height": (4, 8), "height": 10, "width": 10, "depth": 10},
    ],
    [A.Crop, {"x_max": 64, "y_max": 64, "z_max": 64}],
    [A.ToFloat, {"max_value": 16536}],
    [A.Normalize, {"mean": (0.385), "std": (0.129)}],
    [A.RandomScale, {"scale_limit": 0.2, "interpolation": 2}],
    [A.Resize, {"height": 64, "width": 64, "depth": 64}],
    [A.SmallestMaxSize, {"max_size": 64, "interpolation": 2}],
    [A.LongestMaxSize, {"max_size": 128, "interpolation": 2}],
    [A.Posterize, {"num_bits": 8}],
    [A.Equalize, {}],
    [A.Sharpen, {"alpha": [0.2, 0.5], "lightness": [0.5, 1.0]}],
    [
        A.CropAndPad,
        {
            "px": 10,
            "keep_size": False,
            "sample_independently": False,
            "interpolation": 2,
            "pad_cval_mask": 10,
            "pad_cval": 10,
            "pad_mode": "constant",
        },
    ],
    [A.Downscale, dict(scale_min=0.5, scale_max=0.75, interpolation=1)],
    [A.Flip, {}],
    [A.FromFloat, dict(dtype="uint8", max_value=1)],
    [A.HorizontalFlip, {}],
    [A.SliceFlip, {}],
    [A.InvertImg, {}],
    [A.NoOp, {}],
    [A.RandomRotate90, {"axes": "xy"}],
    [A.Transpose, {}],
    [A.VerticalFlip, {}],
    [
        A.UnsharpMask,
        {"blur_limit": 3, "sigma_limit": 0.5, "alpha": 0.2, "threshold": 15},
    ],
    [A.PixelDropout, {"dropout_prob": 0.1, "per_channel": True, "drop_value": None}],
    [
        A.PixelDropout,
        {
            "dropout_prob": 0.1,
            "per_channel": False,
            "drop_value": None,
            "mask_drop_value": 15,
        },
    ],
    [
        A.RandomCropFromBorders,
        dict(
            crop_left=0.2,
            crop_right=0.3,
            crop_top=0.05,
            crop_bottom=0.5,
            crop_close=0.1,
            crop_far=0.8,
        ),
    ],
]

AUGMENTATION_CLS_EXCEPT = {
    A.RandomCropNearBBox,
    A.RandomSizedBBoxSafeCrop,
    A.BBoxSafeRandomCrop,
    A.GridDropout,
    A.RescaleSlopeIntercept,
    A.SetPixelSpacing,
    A.NPSNoise,
}


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    check_all_augs_exists(AUGMENTATION_CLS_PARAMS, AUGMENTATION_CLS_EXCEPT),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization_with_custom_parameters(
    augmentation_cls, params, p, seed, image, mask, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    check_all_augs_exists(AUGMENTATION_CLS_PARAMS, AUGMENTATION_CLS_EXCEPT),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
@pytest.mark.parametrize("data_format", ("yaml",))
def test_augmentations_serialization_to_file_with_custom_parameters(
    augmentation_cls, params, p, seed, image, mask, always_apply, data_format
):
    with patch("builtins.open", OpenMock()):
        aug = augmentation_cls(p=p, always_apply=always_apply, **params)
        filepath = "serialized.{}".format(data_format)
        A.save(aug, filepath, data_format=data_format)
        deserialized_aug = A.load(filepath, data_format=data_format)
        set_seed(seed)
        aug_data = aug(image=image, mask=mask)
        set_seed(seed)
        deserialized_aug_data = deserialized_aug(image=image, mask=mask)
        assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
        assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedBBoxSafeCrop: {"height": 10, "width": 10, "depth": 10},
            A.BBoxSafeRandomCrop: {"erosion_rate": 0.6},
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.CoarseDropout,
            A.GridDropout,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_for_bboxes_serialization(
    augmentation_cls, params, p, seed, image, dicaugment_bboxes, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, bboxes=dicaugment_bboxes)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, bboxes=dicaugment_bboxes)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.LongestMaxSize,
            A.SmallestMaxSize,
            A.PadIfNeeded,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
            A.CoarseDropout,
            A.GridDropout,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
        },
    ),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_for_keypoints_serialization(
    augmentation_cls, params, p, seed, image, keypoints, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, keypoints=keypoints)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, keypoints=keypoints)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["keypoints"], deserialized_aug_data["keypoints"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params", "call_params"],
    [
        [
            A.RandomCropNearBBox,
            {"max_part_shift": 0.15},
            {"cropping_bbox": [-59, 77, 40, 177, 231, 221]},
        ]
    ],
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization_with_call_params(
    augmentation_cls, params, call_params, p, seed, image, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    annotations = {"image": image, **call_params}
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(**annotations)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(**annotations)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])


def test_from_float_serialization(float_image):
    aug = A.FromFloat(p=1, dtype="uint8")
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    aug_data = aug(image=float_image)
    deserialized_aug_data = deserialized_aug(image=float_image)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization(seed, image, mask):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.Resize(128, 128, 128),
                        A.RandomSizedCrop(
                            min_max_height=(32, 64),
                            height=128,
                            width=128,
                            depth=128,
                            p=1,
                        ),
                        A.OneOf(
                            [
                                A.RandomSizedCrop(
                                    min_max_height=(32, 64),
                                    height=100,
                                    width=100,
                                    depth=100,
                                    p=0.5,
                                ),
                                A.RandomSizedCrop(
                                    min_max_height=(32, 64),
                                    height=128,
                                    width=128,
                                    depth=128,
                                    p=0.5,
                                ),
                            ]
                        ),
                    ]
                ),
                A.Compose(
                    [
                        A.Resize(100, 100, 100),
                        A.RandomSizedCrop(
                            min_max_height=(32, 64),
                            height=100,
                            width=100,
                            depth=100,
                            p=1,
                        ),
                    ]
                ),
            ),
            A.SomeOf(
                [
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                2,
                replace=False,
            ),
        ]
    )
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["bboxes", "bbox_format", "labels"],
    [
        ([(20, 30, 40, 40, 50, 50)], "coco_3d", [1]),
        ([(20, 30, 40, 40, 50, 50, 99), (10, 40, 50, 30, 20, 25, 9)], "coco_3d", None),
        ([(20, 30, 40, 60, 80, 90)], "pascal_voc_3d", [2]),
        ([(20, 30, 40, 60, 80, 90, 99)], "pascal_voc_3d", None),
        ([(0.1, 0.2, 0.3, 0.1, 0.2, 0.3)], "yolo_3d", [2]),
        ([(0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 99)], "yolo_3d", None),
    ],
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization_with_bboxes(
    seed, image, bboxes, bbox_format, labels
):
    if labels is None:
        bbox_params = {"format": bbox_format}
    else:
        bbox_params = {"format": bbox_format, "label_fields": ["labels"]}

    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.RandomRotate90(),
                        A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                    ]
                ),
                A.Compose(
                    [
                        A.Rotate(p=0.5),
                    ]
                ),
            ),
            A.SomeOf(
                [
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                n=5,
            ),
        ],
        bbox_params=bbox_params,
    )
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = (
        aug(image=image, bboxes=bboxes)
        if labels is None
        else aug(image=image, bboxes=bboxes, labels=labels)
    )
    set_seed(seed)
    deserialized_aug_data = (
        deserialized_aug(image=image, bboxes=bboxes)
        if labels is None
        else deserialized_aug(image=image, bboxes=bboxes, labels=labels)
    )
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 40, 50, 60)], "xyzas", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "xyz", [1, 2]),
        ([(20, 30, 60, 80)], "zyx", [2]),
        ([(20, 30, 60, 80, 99)], "xyzs", [2]),
    ],
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization_with_keypoints(
    seed, image, keypoints, keypoint_format, labels
):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.RandomRotate90(),
                        A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                    ]
                ),
                A.Compose(
                    [
                        A.Rotate(p=0.5),
                    ]
                ),
            ),
            A.SomeOf(
                n=2,
                transforms=[
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                replace=False,
            ),
        ],
        keypoint_params={"format": keypoint_format, "label_fields": ["labels"]},
    )
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, keypoints=keypoints, labels=labels)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(
        image=image, keypoints=keypoints, labels=labels
    )
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["keypoints"], deserialized_aug_data["keypoints"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        except_augmentations={
            A.RescaleSlopeIntercept,
            A.NPSNoise,
        },
    ),
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_additional_targets_for_image_only_serialization(
    augmentation_cls, params, image, seed
):
    aug = A.Compose(
        [augmentation_cls(always_apply=True, **params)],
        additional_targets={"image2": "image"},
    )
    image2 = image.copy()

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, image2=image2)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, image2=image2)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["image2"], deserialized_aug_data["image2"])


def test_custom_transform_with_overlapping_name():
    class HorizontalFlip(ImageOnlyTransform):
        pass

    assert SERIALIZABLE_REGISTRY["HorizontalFlip"] == A.HorizontalFlip
    assert (
        SERIALIZABLE_REGISTRY["tests.test_serialization.HorizontalFlip"]
        == HorizontalFlip
    )


def test_serialization_v2_to_dict():
    transform = A.Compose([A.HorizontalFlip()])
    transform_dict = A.to_dict(transform)["transform"]
    assert transform_dict == {
        "__class_fullname__": "Compose",
        "p": 1.0,
        "transforms": [
            {"__class_fullname__": "HorizontalFlip", "always_apply": False, "p": 0.5}
        ],
        "bbox_params": None,
        "keypoint_params": None,
        "additional_targets": {},
        "is_check_shapes": True,
    }


@pytest.mark.parametrize(
    ["class_fullname", "expected_short_class_name"],
    [
        ["dicaugment.augmentations.transforms.HorizontalFlip", "HorizontalFlip"],
        ["HorizontalFlip", "HorizontalFlip"],
        ["some_module.HorizontalFlip", "some_module.HorizontalFlip"],
    ],
)
def test_shorten_class_name(class_fullname, expected_short_class_name):
    assert shorten_class_name(class_fullname) == expected_short_class_name
