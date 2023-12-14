import random
from typing import Dict, Tuple, Type

import cv2
import numpy as np
import pytest

import dicaugment as A

from numpy.testing import assert_array_almost_equal_nulp

from .utils import get_dual_transforms, get_image_only_transforms, get_transforms


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={},
        except_augmentations={
            A.FromFloat,
            A.Normalize,
            A.ToFloat,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
def test_image_only_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            # A.HistogramMatching: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            # },
            # A.PixelDistributionAdaptation: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            #     "transform_type": "standard",
            # },
            A.MedianBlur: {"blur_limit": (3, 5)},
            # A.TemplateTransform: {
            #     "templates": np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32),
            # },
            # A.RingingOvershoot: {"blur_limit": (3, 5)},
        },
        except_augmentations={
            # A.CLAHE,
            A.Equalize,
            # A.FancyPCA,
            A.FromFloat,
            # A.ISONoise,
            A.Posterize,
            # A.RandomToneCurve,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
def test_image_only_augmentations_with_float_values(
    augmentation_cls, params, float_image, mask
):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
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
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
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
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
def test_dual_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
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
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
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
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
def test_dual_augmentations_with_float_values(
    augmentation_cls, params, float_image, mask
):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # A.HistogramMatching: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.PixelDistributionAdaptation: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            #     "transform_type": "standard",
            # },
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
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
            # A.TemplateTransform: {
            #     "templates": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            # },
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RescaleSlopeIntercept,
            A.NPSNoise,
            A.SetPixelSpacing,
        },
    ),
)
def test_augmentations_wont_change_input(augmentation_cls, params, image, mask):
    image_copy = image.copy()
    mask_copy = mask.copy()
    aug = augmentation_cls(p=1, **params)
    aug(image=image, mask=mask)
    assert np.array_equal(image, image_copy)
    assert np.array_equal(mask, mask_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # A.HistogramMatching: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            # },
            # A.PixelDistributionAdaptation: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            #     "transform_type": "standard",
            # },
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
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
            # A.TemplateTransform: {
            #     "templates": np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32),
            # },
        },
        except_augmentations={
            # A.CLAHE,
            A.Equalize,
            # A.FancyPCA,
            # A.ISONoise,
            A.Posterize,
            # A.RandomToneCurve,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise
            # A.CropNonEmptyMaskIfExists,
            # A.MaskDropout,
        },
    ),
)
def test_augmentations_wont_change_float_input(augmentation_cls, params, float_image):
    float_image_copy = float_image.copy()
    aug = augmentation_cls(p=1, **params)
    aug(image=float_image)
    assert np.array_equal(float_image, float_image_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # A.HistogramMatching: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.Normalize: {"mean": 0, "std": 1},
            # A.TemplateTransform: {
            #     "templates": np.random.randint(low=0, high=256, size=(224, 224), dtype=np.uint8),
            # },
        },
        except_augmentations={
            # A.ChannelDropout,
            # A.ChannelShuffle,
            # A.FancyPCA,
            # A.ISONoise,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CenterCrop,
            A.Crop,
            # A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            # A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.LongestMaxSize,
            A.SmallestMaxSize,
            A.PadIfNeeded,
            # A.RGBShift,
            # A.RandomFog,
            # A.RandomGravel,
            # A.RandomRain,
            A.RandomScale,
            # A.RandomShadow,
            # A.RandomSnow,
            # A.RandomSunFlare,
            # A.ToRGB,
            # A.ToSepia,
            # A.PixelDistributionAdaptation,
            A.UnsharpMask,
            A.RandomCropFromBorders,
            # A.Spatter,
        },
    ),
)
def test_augmentations_wont_change_shape_grayscale(
    augmentation_cls, params, image, mask
):
    aug = augmentation_cls(p=1, **params)

    # Test for grayscale image
    image = np.zeros((100, 100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100, 100))
    result = aug(image=image, mask=mask)
    assert np.array_equal(image.shape, result["image"].shape)
    assert np.array_equal(mask.shape, result["mask"].shape)

    # Test for grayscale image with dummy dim
    image_1ch = np.zeros((100, 100, 100, 1), dtype=np.int16)
    mask_1ch = np.zeros((100, 100, 100, 1))

    result = aug(image=image_1ch, mask=mask_1ch)
    assert np.array_equal(image_1ch.shape, result["image"].shape)
    assert np.array_equal(mask_1ch.shape, result["mask"].shape)


# @pytest.mark.parametrize(
#     ["augmentation_cls", "params"],
#     get_transforms(
#         custom_arguments={
#             A.HistogramMatching: {
#                 "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
#                 "read_fn": lambda x: x,
#             },
#             A.FDA: {
#                 "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
#                 "read_fn": lambda x: x,
#             },
#             A.PixelDistributionAdaptation: {
#                 "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
#                 "read_fn": lambda x: x,
#                 "transform_type": "standard",
#             },
#             A.TemplateTransform: {
#                 "templates": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
#             },
#         },
#         except_augmentations={
#             A.RandomCropNearBBox,
#             A.RandomSizedBBoxSafeCrop,
#             A.BBoxSafeRandomCrop,
#             A.CenterCrop,
#             A.Crop,
#             A.CropNonEmptyMaskIfExists,
#             A.RandomCrop,
#             A.RandomResizedCrop,
#             A.RandomSizedCrop,
#             A.CropAndPad,
#             A.Resize,
#             A.LongestMaxSize,
#             A.SmallestMaxSize,
#             A.PadIfNeeded,
#             A.RandomScale,
#             A.RandomCropFromBorders,
#         },
#     ),
# )
# def test_augmentations_wont_change_shape_rgb(augmentation_cls, params, image, mask):
#     aug = augmentation_cls(p=1, **params)

#     # Test for RGB image
#     image_3ch = np.zeros((224, 224, 3), dtype=np.uint8)
#     mask_3ch = np.zeros((224, 224, 3))

#     result = aug(image=image_3ch, mask=mask_3ch)
#     assert np.array_equal(image_3ch.shape, result["image"].shape)
#     assert np.array_equal(mask_3ch.shape, result["mask"].shape)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"], [[A.RandomCropNearBBox, {"max_part_shift": 0.15}]]
)
def test_image_only_crop_around_bbox_augmentation(
    augmentation_cls, params, image, mask
):
    aug = augmentation_cls(p=1, **params)
    annotations = {"image": image, "cropping_bbox": [-59, 77, 20, 177, 231, 40]}
    data = aug(**annotations)
    assert data["image"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [
            A.PadIfNeeded,
            {
                "min_height": 128,
                "min_width": 128,
                "min_depth": 128,
                "border_mode": "constant",
                "value": 100,
                "mask_value": 1,
            },
        ],
        [
            A.Rotate,
            {"border_mode": "constant", "axes": "xy", "value": 100, "mask_value": 1},
        ],
        [A.LongestMaxSize, {"max_size": 50}],
        [A.SmallestMaxSize, {"max_size": 50}],
        # [A.SafeRotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [
            A.ShiftScaleRotate,
            {"border_mode": "constant", "value": 100, "mask_value": 1},
        ],
        # [A.OpticalDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        # [A.ElasticTransform, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        # [A.GridDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        # [A.Affine, {"mode": cv2.BORDER_CONSTANT, "cval_mask": 1, "cval": 100}],
        # [A.PiecewiseAffine, {"mode": "constant", "cval_mask": 1, "cval": 100}],
    ],
)
def test_mask_fill_value(augmentation_cls, params):
    random.seed(42)
    aug = augmentation_cls(p=1, **params)
    input = {
        "image": np.zeros((100, 100, 100), dtype=np.int16) + 100,
        "mask": np.ones((100, 100, 100)),
    }
    output = aug(**input)
    assert (output["image"] == 100).all()
    assert np.isclose(output["mask"], 1).all()
    # assert (output["mask"] == 1).all()


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # A.HistogramMatching: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
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
            # A.TemplateTransform: {
            #     "templates": np.random.randint(0, 256, (100, 100, 6), dtype=np.uint8),
            # },
        },
        except_augmentations={
            # A.CLAHE,
            # A.ColorJitter,
            # A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            # A.HueSaturationValue,
            # A.ISONoise,
            # A.ImageCompression,
            # A.MaskDropout,
            A.Normalize,
            # A.RGBShift,
            A.RandomCropNearBBox,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
            # A.RandomFog,
            # A.RandomGravel,
            # A.RandomRain,
            # A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            # A.RandomSnow,
            # A.RandomSunFlare,
            A.ToFloat,
            # A.ToGray,
            # A.ToRGB,
            # A.ToSepia,
            # A.FancyPCA,
            # A.PixelDistributionAdaptation,
            # A.Spatter,
        },
    ),
)
def test_multichannel_image_augmentations(augmentation_cls, params):
    image = np.zeros((50, 50, 50, 6), dtype=np.uint8)
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[3] == 6


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # A.HistogramMatching: {
            #     "reference_images": [np.random.uniform(0.0, 1.0, (100, 100, 6)).astype(np.float32)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.Normalize: {"mean": 0, "std": 1},
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
            # A.TemplateTransform: {
            #     "templates": np.random.uniform(0.0, 1.0, (100, 100, 6)).astype(np.float32),
            # },
        },
        except_augmentations={
            # A.CLAHE,
            # A.ColorJitter,
            # A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            # A.HueSaturationValue,
            # A.ISONoise,
            # A.ImageCompression,
            # A.MaskDropout,
            # A.RGBShift,
            A.RandomCropNearBBox,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
            # A.RandomFog,
            # A.RandomGravel,
            # A.RandomRain,
            # A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            # A.RandomSnow,
            # A.RandomSunFlare,
            # A.ToGray,
            # A.ToRGB,
            # A.ToSepia,
            A.Equalize,
            # A.FancyPCA,
            A.Posterize,
            # A.RandomToneCurve,
            # A.PixelDistributionAdaptation,
            # A.Spatter,
        },
    ),
)
def test_float_multichannel_image_augmentations(augmentation_cls, params):
    image = np.zeros((50, 50, 50, 6), dtype=np.float32)
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.float32
    assert data["image"].shape[3] == 6


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
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
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
            # A.TemplateTransform: {
            #     "templates": np.random.randint(0, 1, (100, 100), dtype=np.uint8),
            # },
        },
        except_augmentations={
            # A.CLAHE,
            # A.ColorJitter,
            # A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            # A.HueSaturationValue,
            # A.ISONoise,
            # A.ImageCompression,
            # A.MaskDropout,
            A.Normalize,
            # A.RGBShift,
            A.RandomCropNearBBox,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
            # A.RandomFog,
            # A.RandomGravel,
            # A.RandomRain,
            # A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            # A.RandomSnow,
            # A.RandomSunFlare,
            A.ToFloat,
            # A.ToGray,
            # A.ToRGB,
            # A.ToSepia,
            # A.FancyPCA,
            # A.FDA,
            # A.HistogramMatching,
            # A.PixelDistributionAdaptation,
            # A.Spatter,
        },
    ),
)
def test_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    for num_channels in range(3, 13):
        image = np.zeros((50, 50, 50, num_channels), dtype=np.int16)
        aug = augmentation_cls(p=1, **params)
        data = aug(image=image)
        assert data["image"].dtype == np.int16
        assert data["image"].shape[3] == num_channels


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
            # A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.Normalize: {"mean": 0, "std": 1},
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
            # A.TemplateTransform: {
            #     "templates": np.random.uniform(0.0, 1.0, (100, 100, 1)).astype(np.float32),
            # },
        },
        except_augmentations={
            # A.CLAHE,
            # A.ColorJitter,
            # A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            # A.HueSaturationValue,
            # A.ISONoise,
            # A.ImageCompression,
            # A.MaskDropout,
            # A.RGBShift,
            A.RandomCropNearBBox,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
            # A.RandomFog,
            # A.RandomGravel,
            # A.RandomRain,
            # A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            # A.RandomSnow,
            # A.RandomSunFlare,
            # A.ToGray,
            # A.ToRGB,
            # A.ToSepia,
            A.Equalize,
            # A.FancyPCA,
            A.Posterize,
            # A.RandomToneCurve,
            # A.FDA,
            # A.HistogramMatching,
            # A.PixelDistributionAdaptation,
            # A.Spatter,
        },
    ),
)
def test_float_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    for num_channels in range(3, 13):
        image = np.zeros((50, 50, 50, num_channels), dtype=np.float32)
        aug = augmentation_cls(p=1, **params)
        data = aug(image=image)
        assert data["image"].dtype == np.float32
        assert data["image"].shape[3] == num_channels


@pytest.mark.parametrize(
    ["augmentation_cls", "params", "image_shape"],
    [
        [
            A.PadIfNeeded,
            {"min_height": 100, "min_width": 100, "min_depth": 100},
            (50, 50, 50),
        ],
        [
            A.PadIfNeeded,
            {"min_height": 100, "min_width": 100, "min_depth": 100},
            (75, 75, 50),
        ],
        [
            A.PadIfNeeded,
            {"min_height": 100, "min_width": 100, "min_depth": 100},
            (128, 128, 128),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": None,
                "min_width": None,
                "min_depth": None,
                "pad_height_divisor": 100,
                "pad_width_divisor": 100,
                "pad_depth_divisor": 100,
            },
            (50, 50, 50),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": None,
                "min_width": None,
                "min_depth": None,
                "pad_height_divisor": 150,
                "pad_width_divisor": 100,
                "pad_depth_divisor": 75,
            },
            (50, 100, 25),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": None,
                "min_width": None,
                "min_depth": None,
                "pad_height_divisor": 75,
                "pad_width_divisor": 20,
                "pad_depth_divisor": 30,
            },
            (15, 15, 15),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": None,
                "min_width": None,
                "min_depth": None,
                "pad_height_divisor": 72,
                "pad_width_divisor": 40,
                "pad_depth_divisor": 36,
            },
            (50, 60, 70),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": None,
                "min_width": None,
                "min_depth": None,
                "pad_height_divisor": 72,
                "pad_width_divisor": 128,
                "pad_depth_divisor": 64,
            },
            (50, 50, 50),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": 100,
                "min_width": None,
                "min_depth": None,
                "pad_width_divisor": 100,
                "pad_depth_divisor": 100,
            },
            (50, 60, 70),
        ],
        [
            A.PadIfNeeded,
            {
                "min_height": None,
                "min_width": 100,
                "min_depth": None,
                "pad_height_divisor": 100,
                "pad_depth_divisor": 100,
            },
            (50, 60, 70),
        ],
    ],
)
def test_pad_if_needed(
    augmentation_cls: Type[A.PadIfNeeded], params: Dict, image_shape: Tuple[int, int]
):
    image = np.zeros(image_shape)
    pad = augmentation_cls(**params)
    image_padded = pad(image=image)["image"]

    if pad.min_width is not None:
        assert image_padded.shape[1] >= pad.min_width

    if pad.min_height is not None:
        assert image_padded.shape[0] >= pad.min_height

    if pad.min_depth is not None:
        assert image_padded.shape[2] >= pad.min_depth

    if pad.pad_width_divisor is not None:
        assert image_padded.shape[1] % pad.pad_width_divisor == 0
        assert image_padded.shape[1] >= image.shape[1]
        assert image_padded.shape[1] - image.shape[1] <= pad.pad_width_divisor

    if pad.pad_height_divisor is not None:
        assert image_padded.shape[0] % pad.pad_height_divisor == 0
        assert image_padded.shape[0] >= image.shape[0]
        assert image_padded.shape[0] - image.shape[0] <= pad.pad_height_divisor

    if pad.pad_depth_divisor is not None:
        assert image_padded.shape[2] % pad.pad_depth_divisor == 0
        assert image_padded.shape[2] >= image.shape[2]
        assert image_padded.shape[2] - image.shape[2] <= pad.pad_depth_divisor


@pytest.mark.parametrize(
    ["params", "image_shape"],
    [
        [
            {
                "min_height": 10,
                "min_width": 12,
                "min_depth": 10,
                "border_mode": "constant",
                "value": 1,
                "position": "center",
            },
            (5, 6, 5),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "min_depth": 10,
                "border_mode": "constant",
                "value": 1,
                "position": "front_top_left",
            },
            (5, 6, 5),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "min_depth": 10,
                "border_mode": "constant",
                "value": 1,
                "position": "front_top_right",
            },
            (5, 6, 5),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "min_depth": 10,
                "border_mode": "constant",
                "value": 1,
                "position": "front_bottom_left",
            },
            (5, 6, 5),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "min_depth": 10,
                "border_mode": "constant",
                "value": 1,
                "position": "front_bottom_right",
            },
            (5, 6, 5),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "min_depth": 10,
                "border_mode": "constant",
                "value": 1,
                "position": "random",
            },
            (5, 6, 5),
        ],
    ],
)
def test_pad_if_needed_position(params, image_shape):
    random.seed(42)

    image = np.zeros(image_shape)
    pad = A.PadIfNeeded(**params)
    image_padded = pad(image=image)["image"]

    true_result = np.ones(
        (
            max(image_shape[0], params["min_height"]),
            max(image_shape[1], params["min_width"]),
            max(image_shape[2], params["min_depth"]),
        )
    )

    if params["position"] == "center":
        x_start = image_shape[0] // 2
        y_start = image_shape[1] // 2
        z_start = image_shape[2] // 2
        true_result[
            x_start : x_start + image_shape[0],
            y_start : y_start + image_shape[1],
            z_start : z_start + image_shape[2],
        ] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "front_top_left":
        true_result[: image_shape[0], : image_shape[1], : image_shape[2]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "front_top_right":
        true_result[: image_shape[0], -image_shape[1] :, : image_shape[2]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "front_bottom_left":
        true_result[-image_shape[0] :, : image_shape[1], : image_shape[2]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "front_bottom_right":
        true_result[-image_shape[0] :, -image_shape[1] :, : image_shape[2]] = 0
        assert (image_padded == true_result).all()

    # Not set for seed
    # elif params["position"] == "random":
    #     true_result[0:5, -7:-1, 0:5] = 0
    #     assert (image_padded == true_result).all()


# @pytest.mark.parametrize(
#     ["points"],
#     [
#         [
#             [
#                 [37.25756906, 11.0567457],
#                 [514.03919117, 9.49484312],
#                 [585.66154354, 74.97413793],
#                 [63.60979494, 85.39815904],
#             ]
#         ],
#         [
#             [
#                 [37, 11],
#                 [514, 9],
#                 [585, 74],
#                 [63, 85],
#             ]
#         ],
#         [
#             [
#                 [10, 20],
#                 [719, 34],
#                 [613, 63],
#                 [91, 33],
#             ]
#         ],
#     ],
# )
# def test_perspective_order_points(points):
#     points = np.array(points)
#     res = A.Perspective._order_points(points)
#     assert len(points) == len(np.unique(res, axis=0))


# @pytest.mark.parametrize(
#     ["seed", "scale", "h", "w"],
#     [
#         [0, 0.08, 89, 628],
#         [0, 0.15, 89, 628],
#         [0, 0.15, 35, 190],
#     ],
# )
# def test_perspective_valid_keypoints_after_transform(seed: int, scale: float, h: int, w: int):
#     random.seed(seed)
#     np.random.seed(seed)

#     image = np.zeros([h, w, 3], dtype=np.uint8)
#     keypoints = [
#         [0, 0],
#         [0, h - 1],
#         [w - 1, h - 1],
#         [w - 1, 0],
#     ]

#     transform = A.Compose(
#         [A.Perspective(scale=(scale, scale), p=1)], keypoint_params={"format": "xy", "remove_invisible": False}
#     )

#     res = transform(image=image, keypoints=keypoints)["keypoints"]

#     x1, y1 = res[0]
#     x2, y2 = res[1]
#     x3, y3 = res[2]
#     x4, y4 = res[3]

#     assert x1 < x3 and x1 < x4 and x2 < x3 and x2 < x4 and y1 < y2 and y1 < y3 and y4 < y2 and y4 < y3


# @pytest.mark.parametrize("kind", ["pca", "minmax", "standard"])
# def test_pixel_domain_adaptation(kind):
#     img_uint8 = np.random.randint(low=100, high=200, size=(100, 100, 3), dtype=np.uint8)
#     ref_img_uint8 = np.random.randint(low=0, high=100, size=(100, 100, 3), dtype=np.uint8)
#     img_float, ref_img_float = [x.astype("float32") / 255.0 for x in (img_uint8, ref_img_uint8)]

#     for img, ref_img in ((img_uint8, ref_img_uint8), (img_float, ref_img_float)):
#         adapter = A.PixelDistributionAdaptation(
#             reference_images=[ref_img],
#             blend_ratio=(1, 1),
#             read_fn=lambda x: x,
#             always_apply=True,
#             transform_type=kind,
#         )
#         adapted = adapter(image=img)["image"]
#         np.testing.assert_allclose(
#             adapted.mean(),
#             ref_img.mean(),
#             rtol=0,
#             atol=2 if img.dtype == np.uint8 else 0.01,
#             err_msg=f"{adapted.mean()} {img.mean()} {ref_img.mean()}",
#         )


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # only image
            # A.HistogramMatching: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.FDA: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.uint8)],
            #     "read_fn": lambda x: x,
            # },
            # A.PixelDistributionAdaptation: {
            #     "reference_images": [np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.uint8)],
            #     "read_fn": lambda x: x,
            #     "transform_type": "standard",
            # },
            A.MedianBlur: {"blur_limit": (3, 5)},
            # A.TemplateTransform: {
            #     "templates": np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.uint8),
            # },
            # A.RingingOvershoot: {"blur_limit": (3, 5)},
            # dual
            A.Crop: {
                "y_min": 0,
                "y_max": 10,
                "x_min": 0,
                "x_max": 10,
                "z_min": 0,
                "z_max": 10,
            },
            A.CenterCrop: {"height": 10, "width": 10, "depth": 10},
            # A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            # A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedBBoxSafeCrop: {"height": 10, "width": 10, "depth": 10},
            A.BBoxSafeRandomCrop: {"erosion_rate": 0.5},
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
        },
        except_augmentations=[A.RescaleSlopeIntercept, A.SetPixelSpacing, A.NPSNoise],
    ),
)
def test_non_contiguous_input(augmentation_cls, params, bboxes):
    image = np.empty([100, 100, 100], dtype=np.int16).transpose(1, 2, 0)
    mask = np.empty([100, 100, 100], dtype=np.int16).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]
    assert not mask.flags["C_CONTIGUOUS"]

    if augmentation_cls == A.RandomCropNearBBox:
        # requires "cropping_bbox" arg
        aug = augmentation_cls(p=1, **params)
        aug(image=image, mask=mask, cropping_bbox=bboxes[0])
    elif augmentation_cls in [A.RandomSizedBBoxSafeCrop, A.BBoxSafeRandomCrop]:
        # requires "bboxes" arg
        aug = A.Compose(
            [augmentation_cls(p=1, **params)],
            bbox_params=A.BboxParams(format="pascal_voc_3d"),
        )
        aug(image=image, mask=mask, bboxes=bboxes)
    else:
        # standard args: image and mask
        if augmentation_cls == A.FromFloat:
            # requires float image
            max_value = A.MAX_VALUES_BY_DTYPE[image.dtype]
            min_value = A.MIN_VALUES_BY_DTYPE[image.dtype]
            image = (image.astype("float32") - min_value) / (max_value - min_value)
            assert not image.flags["C_CONTIGUOUS"]

        aug = augmentation_cls(p=1, **params)
        aug(image=image, mask=mask)

    # OK, if no exception is raised
