import random
from functools import partial

import cv2
import numpy as np
import pytest

import dicaugment as A
import dicaugment.augmentations.functional as F
import dicaugment.augmentations.geometric.functional as FGeometric
from dicaugment.augmentations.blur.functional import gaussian_blur

from .utils import get_dual_transforms, get_image_only_transforms, get_transforms


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6, 3))
    augmentation = A.Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented["image"].shape == (6, 8, 3)
    assert augmented["mask"].shape == (6, 8, 3)


@pytest.mark.parametrize(
    "interpolation", [A.INTER_NEAREST, A.INTER_LINEAR, A.INTER_CUBIC]
)
def test_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 10), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100, 10), dtype=np.uint8)
    aug = A.Rotate(limit=(45, 45), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.rotate(
        image, 45, axes="xy", interpolation=interpolation, border_mode="constant"
    )
    expected_mask = FGeometric.rotate(
        mask, 45, axes="xy", interpolation=A.INTER_NEAREST, border_mode="constant"
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


def test_rotate_crop_border():
    image = np.random.randint(low=100, high=256, size=(100, 100, 10), dtype=np.uint8)
    border_value = 13
    aug = A.Rotate(
        limit=(45, 45),
        p=1,
        value=border_value,
        border_mode="constant",
        crop_to_border=True,
    )
    aug_img = aug(image=image)["image"]
    expected_size = int(np.round(100 / np.sqrt(2))) * 2
    assert aug_img.shape[0] == expected_size


@pytest.mark.parametrize(
    "interpolation", [A.INTER_NEAREST, A.INTER_LINEAR, A.INTER_CUBIC]
)
def test_shift_scale_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 10), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100, 10), dtype=np.uint8)
    aug = A.ShiftScaleRotate(
        shift_limit=(0.2, 0.2),
        axes="xy",
        scale_limit=(1.1, 1.1),
        rotate_limit=(45, 45),
        interpolation=interpolation,
        p=1,
    )
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.shift_scale_rotate(
        image,
        angle=45,
        axes="xy",
        scale=2.1,
        dx=0.2,
        dy=0.2,
        dz=0.2,
        interpolation=interpolation,
        border_mode="constant",
    )
    expected_mask = FGeometric.shift_scale_rotate(
        mask,
        angle=45,
        axes="xy",
        scale=2.1,
        dx=0.2,
        dy=0.2,
        dz=0.2,
        interpolation=A.INTER_NEAREST,
        border_mode="constant",
    )
    assert np.isclose(data["image"], expected_image).all()
    assert np.array_equal(data["mask"], expected_mask)

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
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.PixelDropout: {
                "dropout_prob": 0.5,
                "mask_drop_value": 10,
                "drop_value": 20,
            },
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.PixelDropout,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
def test_binary_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 10), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100, 10), dtype=np.uint8)
    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data["mask"]), np.array([0, 1]))


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
            A.RandomCrop: {"height": 10, "width": 10, "depth": 10},
            A.RandomSizedCrop: {
                "min_max_height": (4, 8),
                "height": 10,
                "width": 10,
                "depth": 10,
            },
            A.Resize: {"height": 10, "width": 10, "depth": 10},
            A.PixelDropout: {
                "dropout_prob": 0.5,
                "mask_drop_value": 10,
                "drop_value": 20,
            },
            A.PadIfNeeded: {"min_height": 100, "min_width": 100, "min_depth": 100},
            A.LongestMaxSize: {"max_size": 50},
            A.SmallestMaxSize: {"max_size": 50},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropAndPad,
            A.PixelDropout,
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise,
        },
    ),
)
def test_semantic_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts.
    Note: IAAAffine, IAAPiecewiseAffine, IAAPerspective does not properly operate if mask has values other than {0;1}
    """
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 10), dtype=np.uint8)
    mask = np.random.randint(low=0, high=4, size=(100, 100, 10), dtype=np.uint8) * 64

    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data["mask"]), np.array([0, 64, 128, 192]))


def __test_multiprocessing_support_proc(args):
    x, transform = args
    return transform(image=x)


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
            A.RescaleSlopeIntercept,
            A.SetPixelSpacing,
            A.NPSNoise
        },
    ),
)
def test_multiprocessing_support(mp_pool, augmentation_cls, params):
    """Checks whether we can use augmentations in multiprocessing environments"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 100), dtype=np.uint8)

    mp_pool.map(
        __test_multiprocessing_support_proc, map(lambda x: (x, aug), [image] * 10)
    )


def test_force_apply():
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
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
                                    min_max_height=(16, 32),
                                    height=64,
                                    width=64,
                                    depth=64,
                                    p=0.5,
                                ),
                                A.RandomSizedCrop(
                                    min_max_height=(32, 64),
                                    height=100,
                                    width=100,
                                    depth=100,
                                    p=0.5,
                                ),
                            ]
                        ),
                    ]
                ),
                A.Compose(
                    [
                        A.RandomSizedCrop(
                            min_max_height=(16, 64), height=64, width=64, depth=64, p=1
                        ),
                        # A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ]
                ),
            ),
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )

    res = aug(image=np.zeros((100, 100, 100), dtype=np.uint8))
    assert res["image"].shape[0] in (64, 100, 128)
    assert res["image"].shape[1] in (64, 100, 128)
    assert res["image"].shape[2] in (64, 100, 128)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={},
        except_augmentations=[A.RescaleSlopeIntercept, A.SetPixelSpacing, A.NPSNoise],
    ),
)
def test_additional_targets_for_image_only(augmentation_cls, params):
    aug = A.Compose(
        [augmentation_cls(always_apply=True, **params)],
        additional_targets={"image2": "image"},
    )
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(50, 50, 10), dtype=np.uint8)
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res["image"]
        aug2 = res["image2"]
        assert np.array_equal(aug1, aug2)


def test_image_invert():
    for _ in range(10):
        # test for np.uint8 dtype
        image1 = np.random.randint(low=0, high=256, size=(50, 50, 10), dtype=np.uint8)
        image2 = A.to_float(image1)
        r_int = F.invert(F.invert(image1))
        r_float = F.invert(F.invert(image2))
        r_to_float = A.to_float(r_int)
        assert np.allclose(r_float, r_to_float, atol=0.01)


@pytest.mark.parametrize(
    "interpolation", [A.INTER_NEAREST, A.INTER_LINEAR, A.INTER_CUBIC]
)
def test_downscale(interpolation):
    img_float = np.random.rand(50, 50, 10)
    img_uint = (img_float * 255).astype("uint8")

    aug = A.Downscale(
        scale_min=0.5, scale_max=0.5, interpolation=interpolation, always_apply=True
    )

    for img in (img_float, img_uint):
        transformed = aug(image=img)["image"]
        func_applied = F.downscale(
            img,
            scale=0.5,
            down_interpolation=interpolation,
            up_interpolation=interpolation,
        )
        np.testing.assert_almost_equal(transformed, func_applied)


def test_crop_keypoints():
    image = np.random.randint(0, 256, (100, 100, 100), np.uint8)
    keypoints = [(50, 50, 50, 0, 0)]

    aug = A.Crop(0, 0, 0, 80, 80, 80, p=1)
    result = aug(image=image, keypoints=keypoints)
    assert result["keypoints"] == keypoints

    aug = A.Crop(50, 50, 50, 100, 100, 100, p=1)
    result = aug(image=image, keypoints=keypoints)
    assert result["keypoints"] == [(0, 0, 0, 0, 0)]


def test_longest_max_size_keypoints():
    img = np.random.randint(0, 256, [50, 10, 10], np.uint8)
    keypoints = [(9, 5, 5, 0, 0)]

    aug = A.LongestMaxSize(max_size=100, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(18, 10, 10, 0, 0)]

    aug = A.LongestMaxSize(max_size=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(0.9, 0.5, 0.5, 0, 0)]

    aug = A.LongestMaxSize(max_size=50, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(9, 5, 5, 0, 0)]


def test_smallest_max_size_keypoints():
    img = np.random.randint(0, 256, [50, 10, 10], np.uint8)
    keypoints = [(9, 5, 5, 0, 0)]

    aug = A.SmallestMaxSize(max_size=100, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(90, 50, 50, 0, 0)]

    aug = A.SmallestMaxSize(max_size=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(4.5, 2.5, 2.5, 0, 0)]

    aug = A.SmallestMaxSize(max_size=10, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(9, 5, 5, 0, 0)]


def test_resize_keypoints():
    img = np.random.randint(0, 256, [50, 10, 10], np.uint8)
    keypoints = [(9, 5, 9, 0, 0)]

    aug = A.Resize(height=100, width=5, depth=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(4.5, 10, 4.5, 0, 0)]

    aug = A.Resize(height=50, width=10, depth=10, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(9, 5, 9, 0, 0)]


@pytest.mark.parametrize(
    "image",
    [
        np.random.randint(0, 256, [50, 50, 50], np.uint8),
        np.random.random([50, 50, 50]).astype(np.float32),
    ],
)
def test_grid_dropout_mask(image):
    mask = np.ones([50, 50, 50], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=0)
    result = aug(image=image, mask=mask)
    # with mask on ones and fill_value = 0 the sum of pixels is smaller
    assert result["image"].sum() < image.sum()
    assert result["image"].shape == image.shape
    assert result["mask"].sum() < mask.sum()
    assert result["mask"].shape == mask.shape

    # with mask of zeros and fill_value = 0 mask should not change
    mask = np.zeros([50, 50, 50], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=0)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert np.all(result["mask"] == 0)

    # with mask mask_fill_value=100, mask sum is larger
    mask = np.random.randint(0, 10, [50, 50, 50], np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=100)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()  # not deterministic
    assert result["mask"].sum() > mask.sum()

    # with mask mask_fill_value=None, mask is not changed
    mask = np.ones([50, 50, 50], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=None)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert result["mask"].sum() == mask.sum()


@pytest.mark.parametrize(
    [
        "ratio",
        "holes_number_x",
        "holes_number_y",
        "holes_number_z",
        "unit_size_min",
        "unit_size_max",
        "shift_x",
        "shift_y",
        "shift_z",
    ],
    [
        (0.00001, 10, 10, 10, 10, 10, 5, 5, 5),
        (0.9, 10, None, 10, 20, None, 0, 0, 0),
        (0.4556, 10, 20, 10, None, 20, 0, 0, 0),
        (0.00004, None, None, None, 2, 10, 0, 0, 0),
    ],
)
def test_grid_dropout_params(
    ratio,
    holes_number_x,
    holes_number_y,
    holes_number_z,
    unit_size_min,
    unit_size_max,
    shift_x,
    shift_y,
    shift_z,
):
    img = np.random.randint(0, 256, [100, 100, 100], np.uint8)

    aug = A.GridDropout(
        ratio=ratio,
        unit_size_min=unit_size_min,
        unit_size_max=unit_size_max,
        holes_number_x=holes_number_x,
        holes_number_y=holes_number_y,
        holes_number_z=holes_number_z,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_z=shift_z,
        random_offset=False,
        fill_value=0,
        p=1,
    )
    result = aug(image=img)["image"]
    # with fill_value = 0 the sum of pixels is smaller
    assert result.sum() < img.sum()
    assert result.shape == img.shape
    params = aug.get_params_dependent_on_targets({"image": img})
    holes = params["holes"]
    assert len(holes[0]) == 6
    # check grid offsets
    if shift_x:
        assert holes[0][0] == shift_x
    else:
        assert holes[0][0] == 0
    if shift_y:
        assert holes[0][1] == shift_y
    else:
        assert holes[0][1] == 0
    if shift_z:
        assert holes[0][2] == shift_z
    else:
        assert holes[0][2] == 0
    # for grid set with limits
    if unit_size_min and unit_size_max:
        assert (
            max(1, unit_size_min * ratio)
            <= (holes[0][3] - holes[0][0])
            <= min(max(1, unit_size_max * ratio), 256)
        )
    elif holes_number_x and holes_number_y:
        assert (holes[0][3] - holes[0][0]) == max(1, int(ratio * 100 // holes_number_x))
        assert (holes[0][4] - holes[0][1]) == max(1, int(ratio * 100 // holes_number_y))
        assert (holes[0][5] - holes[0][2]) == max(1, int(ratio * 100 // holes_number_z))


def test_gauss_noise_incorrect_var_limit_type():
    with pytest.raises(TypeError) as exc_info:
        A.GaussNoise(var_limit={"low": 70, "high": 90})
    message = "Expected var_limit type to be one of (int, float, tuple, list), got <class 'dict'>"
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ["blur_limit", "sigma", "result_blur", "result_sigma"],
    [
        [[0, 0], [1, 1], 0, 1],
        [[1, 1], [0, 0], 1, 0],
        [[1, 1], [1, 1], 1, 1],
        [[0, 0], [0, 0], 3, 0],
        [[0, 3], [0, 0], 3, 0],
        [[0, 3], [0.1, 0.1], 3, 0.1],
    ],
)
def test_gaus_blur_limits(blur_limit, sigma, result_blur, result_sigma):
    img = np.zeros([100, 100, 100], dtype=np.uint8)

    aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma, p=1)])

    res = aug(image=img)["image"]
    assert np.allclose(res, gaussian_blur(img, result_blur, result_sigma))


@pytest.mark.parametrize(
    ["blur_limit", "sigma", "result_blur", "result_sigma"],
    [
        [[0, 0], [1, 1], 0, 1],
        [[1, 1], [0, 0], 1, 0],
        [[1, 1], [1, 1], 1, 1],
    ],
)
def test_unsharp_mask_limits(blur_limit, sigma, result_blur, result_sigma):
    img = np.zeros([100, 100, 100], dtype=np.uint8)

    aug = A.Compose([A.UnsharpMask(blur_limit=blur_limit, sigma_limit=sigma, p=1)])

    res = aug(image=img)["image"]
    assert np.allclose(res, F.unsharp_mask(img, result_blur, result_sigma))


@pytest.mark.parametrize(["val_uint8"], [[0], [1], [128], [255]])
def test_unsharp_mask_float_uint8_diff_less_than_two(val_uint8):
    x_uint8 = np.zeros((5, 5, 5)).astype(np.uint8)
    x_uint8[2, 2, 2] = val_uint8

    x_float32 = np.zeros((5, 5, 5)).astype(np.float32)
    x_float32[2, 2, 2] = val_uint8 / 255.0

    unsharpmask = A.UnsharpMask(blur_limit=3, always_apply=True, p=1)

    random.seed(0)
    usm_uint8 = unsharpmask(image=x_uint8)["image"]

    random.seed(0)
    usm_float32 = unsharpmask(image=x_float32)["image"]

    # Before comparison, rescale the usm_float32 to [0, 255]
    diff = np.abs(usm_uint8 - usm_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


def test_shift_scale_separate_shift_x_shift_y_shift_z(image, mask):
    aug = A.ShiftScaleRotate(
        shift_limit=(0.3, 0.3),
        shift_limit_y=(0.4, 0.4),
        shift_limit_z=(0.5, 0.5),
        scale_limit=0,
        rotate_limit=0,
        p=1,
    )
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.shift_scale_rotate(
        image,
        angle=0,
        scale=1,
        dx=0.3,
        dy=0.4,
        dz=0.5,
        interpolation=A.INTER_LINEAR,
        border_mode="constant",
    )
    expected_mask = FGeometric.shift_scale_rotate(
        mask,
        angle=0,
        scale=1,
        dx=0.3,
        dy=0.4,
        dz=0.5,
        interpolation=A.INTER_NEAREST,
        border_mode="constant",
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


def test_longest_max_size_list():
    img = np.random.randint(0, 256, [50, 10, 10], np.uint8)
    keypoints = [(9, 5, 5, 0, 0)]

    aug = A.LongestMaxSize(max_size=[5, 10], p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["image"].shape in [(10, 2, 2), (5, 1, 1)]
    assert result["keypoints"] in [[(0.9, 0.5, 0.5, 0, 0)], [(1.8, 1, 1, 0, 0)]]


def test_smallest_max_size_list():
    img = np.random.randint(0, 256, [50, 10, 10], np.uint8)
    keypoints = [(9, 5, 5, 0, 0)]

    aug = A.SmallestMaxSize(max_size=[50, 100], p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["image"].shape in [(250, 50, 50), (500, 100, 100)]
    assert result["keypoints"] in [[(45, 25, 25, 0, 0)], [(90, 50, 50, 0, 0)]]


@pytest.mark.parametrize(
    "get_transform",
    [
        # lambda sign: A.Affine(translate_px=sign * 2),
        lambda sign: A.ShiftScaleRotate(
            shift_limit=(sign * 0.02, sign * 0.02), scale_limit=0, rotate_limit=0
        ),
    ],
)
@pytest.mark.parametrize(
    ["bboxes", "expected", "min_volume_visibility", "sign"],
    [
        [[(0, 0, 0, 10, 10, 10, 1)], [], 0.9, -1],
        [[(0, 0, 0, 10, 10, 10, 1)], [(0, 0, 0, 8, 8, 8, 1)], 0.5, -1],
        [[(90, 90, 90, 100, 100, 100, 1)], [], 0.9, 1],
        [[(90, 90, 90, 100, 100, 100, 1)], [(92, 92, 92, 100, 100, 100, 1)], 0.5, 1],
    ],
)
def test_bbox_clipping(
    get_transform, image, bboxes, expected, min_volume_visibility: float, sign: int
):
    transform = get_transform(sign)
    transform.p = 1
    transform = A.Compose(
        [transform],
        bbox_params=A.BboxParams(
            format="pascal_voc_3d", min_volume_visibility=min_volume_visibility
        ),
    )

    res = transform(image=image, bboxes=bboxes)["bboxes"]
    assert np.isclose(res, expected).all()
