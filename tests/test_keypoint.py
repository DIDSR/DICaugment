import math

import numpy as np
import pytest

import albumentations3d as A
import albumentations3d.augmentations.geometric.functional as FGeometric
from albumentations3d.core.keypoints_utils import (
    angle_to_2pi_range,
    convert_keypoint_from_albumentations,
    convert_keypoint_to_albumentations,
    convert_keypoints_from_albumentations,
    convert_keypoints_to_albumentations,
)


@pytest.mark.parametrize(
    ["kp", "source_format", "expected"],
    [
        ((20, 30, 40), "xyz", (20, 30, 40, 0, 0)),
        (np.array([20, 30, 40]), "xyz", (20, 30, 40, 0, 0)),
        ((40, 20, 30), "zyx", (30, 20, 40, 0, 0)),
        ((20, 30, 40, 60), "xyzs", (20, 30, 40, 0, 60)),
        ((20, 30, 40, 60), "xyza", (20, 30, 40, math.radians(60), 0)),
        ((20, 30, 40, 60, 80), "xyzas", (20, 30, 40, math.radians(60), 80)),
    ],
)
def test_convert_keypoint_to_albumentations(kp, source_format, expected):
    image = np.ones((100, 100, 100))

    converted_keypoint = convert_keypoint_to_albumentations(
        kp, rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], source_format=source_format
    )
    assert converted_keypoint == expected


@pytest.mark.parametrize(
    ["kp", "target_format", "expected"],
    [
        ((20, 30, 40, 0, 0), "xyz", (20, 30, 40)),
        ((20, 30, 40, 0, 0), "zyx", (40, 30, 20)),
        ((20, 30, 40, 0.6, 0), "xyza", (20, 30, 40, math.degrees(0.6))),
        ((20, 30, 40, 0, 0.6), "xyzs", (20, 30, 40, 0.6)),
        ((20, 30, 40, 0.6, 80), "xyzas", (20, 30, 40, math.degrees(0.6), 80)),
    ],
)
def test_convert_keypoint_from_albumentations(kp, target_format, expected):
    image = np.ones((100, 100, 100))
    converted_keypoint = convert_keypoint_from_albumentations(
        kp, rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], target_format=target_format
    )
    assert converted_keypoint == expected


@pytest.mark.parametrize(
    ["kp", "keypoint_format"],
    [
        ((20, 30, 40, 50, 60), "xyz"),
        ((20, 30, 40, 50, 60, 99), "xyzas"),
        ((20, 30, 60, 60, 80), "xyzsa"),
        ((20, 30, 60, 80, 99), "zyx"),
    ],
)
def test_convert_keypoint_to_albumentations_and_back(kp, keypoint_format):
    image = np.ones((100, 100, 100))
    converted_kp = convert_keypoint_to_albumentations(
        kp, rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], source_format=keypoint_format
    )
    converted_back_kp = convert_keypoint_from_albumentations(
        converted_kp, rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], target_format=keypoint_format
    )
    assert converted_back_kp == kp


def test_convert_keypoints_to_albumentations():
    keypoints = [(20, 30, 40, 50, 60), (30, 40, 50, 60, 70, 99)]
    image = np.ones((100, 100, 100))
    converted_keypoints = convert_keypoints_to_albumentations(
        keypoints, rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], source_format="xyzas"
    )
    converted_keypoint_1 = convert_keypoint_to_albumentations(
        keypoints[0], rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], source_format="xyzas"
    )
    converted_keypoint_2 = convert_keypoint_to_albumentations(
        keypoints[1], rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], source_format="xyzas"
    )
    assert converted_keypoints == [converted_keypoint_1, converted_keypoint_2]


def test_convert_keypoints_from_albumentations():
    keypoints = [(0.2, 0.3, 0.6, 0.8, 0.85), (0.3, 0.4, 0.7, 0.9, 0.2, 99)]
    image = np.ones((100, 100, 100))
    converted_keypointes = convert_keypoints_from_albumentations(
        keypoints, rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], target_format="xyzas"
    )
    converted_keypoint_1 = convert_keypoint_from_albumentations(
        keypoints[0], rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], target_format="xyzas"
    )
    converted_keypoint_2 = convert_keypoint_from_albumentations(
        keypoints[1], rows=image.shape[0], cols=image.shape[1], slices=image.shape[2], target_format="xyzas"
    )
    assert converted_keypointes == [converted_keypoint_1, converted_keypoint_2]


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 40, 50, 60)], "xyzas", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "xyz", None),
        ([(20, 30, 60, 80)], "zyx", [2]),
        ([(20, 30, 60, 80, 99)], "xyzs", None),
    ],
)
def test_compose_with_keypoint_noop(keypoints, keypoint_format, labels):
    image = np.ones((100, 100, 100))
    if labels is not None:
        aug = A.Compose([A.NoOp(p=1.0)], keypoint_params={"format": keypoint_format, "label_fields": ["labels"]})
        transformed = aug(image=image, keypoints=keypoints, labels=labels)
    else:
        aug = A.Compose([A.NoOp(p=1.0)], keypoint_params={"format": keypoint_format})
        transformed = aug(image=image, keypoints=keypoints)
    assert np.array_equal(transformed["image"], image)
    assert transformed["keypoints"] == keypoints


@pytest.mark.parametrize(["keypoints", "keypoint_format"], [[[[20, 30, 40, 50, 60]], "xyzas"]])
def test_compose_with_keypoint_noop_error_label_fields(keypoints, keypoint_format):
    image = np.ones((100, 100, 100))
    aug = A.Compose([A.NoOp(p=1.0)], keypoint_params={"format": keypoint_format, "label_fields": "class_id"})
    with pytest.raises(Exception):
        aug(image=image, keypoints=keypoints, cls_id=[0])


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 60, 80)], "xyz", {"label": [1]}),
        ([], "xyz", {}),
        ([], "xyz", {"label": []}),
        ([(20, 30, 60, 80)], "xyz", {"id": [3]}),
        ([(20, 30, 60, 80), (30, 40, 40, 50)], "xyz", {"id": [3, 1]}),
    ],
)
def test_compose_with_keypoint_noop_label_outside(keypoints, keypoint_format, labels):
    image = np.ones((100, 100, 100))
    aug = A.Compose([A.NoOp(p=1.0)], keypoint_params={"format": keypoint_format, "label_fields": list(labels.keys())})
    transformed = aug(image=image, keypoints=keypoints, **labels)
    assert np.array_equal(transformed["image"], image)
    assert transformed["keypoints"] == keypoints
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size():
    image = np.ones((100, 100, 100))
    keypoints = [(0.2, 0.3, 0.6, 45, 0), (0.3, 0.4, 0.7, 45, 0, 99)]
    aug = A.RandomSizedCrop(min_max_height=(70, 90), height=50, width=50, depth=50, p=1.0)
    transformed = aug(image=image, keypoints=keypoints)
    assert transformed["image"].shape == (50, 50, 50)
    assert len(keypoints) == len(transformed["keypoints"])


# def test_random_resized_crop_size():
#     image = np.ones((100, 100, 3))
#     keypoints = [(0.2, 0.3, 0.6, 0.8), (0.3, 0.4, 0.7, 0.9, 99)]
#     aug = A.RandomResizedCrop(height=50, width=50, p=1.0)
#     transformed = aug(image=image, keypoints=keypoints)
#     assert transformed["image"].shape == (50, 50, 3)
#     assert len(keypoints) == len(transformed["keypoints"])


@pytest.mark.parametrize(
    ["aug", "keypoints", "expected"],
    [
        [A.HorizontalFlip, [[0, 0, 2]], [[2, 0, 2]]],
        [A.HorizontalFlip, [[2, 0, 0]], [[0, 0, 0]]],
        [A.HorizontalFlip, [[0, 2, 0]], [[2, 2, 0]]],
        [A.HorizontalFlip, [[2, 2, 0]], [[0, 2, 0]]],
        #
        [A.VerticalFlip, [[0, 0, 2]], [[0, 2, 2]]],
        [A.VerticalFlip, [[2, 0, 0]], [[2, 2, 0]]],
        [A.VerticalFlip, [[0, 2, 0]], [[0, 0, 0]]],
        [A.VerticalFlip, [[2, 2, 0]], [[2, 0, 0]]],
        #
        [A.SliceFlip, [[0, 0, 2]], [[0, 0, 0]]],
        [A.SliceFlip, [[2, 0, 2]], [[2, 0, 0]]],
        [A.SliceFlip, [[0, 2, 0]], [[0, 2, 2]]],
        [A.SliceFlip, [[2, 2, 0]], [[2, 2, 2]]],
        #
        [A.HorizontalFlip, [[1, 1, 1]], [[1, 1, 1]]],
        [A.VerticalFlip, [[1, 1, 1]], [[1, 1, 1]]],
        [A.SliceFlip, [[1, 1, 1]], [[1, 1, 1]]],
    ],
)
def test_keypoint_flips_transform_3x3(aug, keypoints, expected):
    transform = A.Compose([aug(p=1)], keypoint_params={"format": "xyz"})

    image = np.ones((3, 3, 3))
    transformed = transform(image=image, keypoints=keypoints, labels=np.ones(len(keypoints)))
    assert np.allclose(expected, transformed["keypoints"])


@pytest.mark.parametrize(
    ["aug", "keypoints", "expected"],
    [
        [A.HorizontalFlip, [[20, 30, 40, 0, 0]], [[79, 30, 40, 180, 0]]],
        [A.HorizontalFlip, [[20, 30, 40, 45, 0]], [[79, 30, 40, 135, 0]]],
        [A.HorizontalFlip, [[20, 30, 40, 90, 0]], [[79, 30, 40, 90, 0]]],
        #
        [A.VerticalFlip, [[20, 30, 40, 0, 0]], [[20, 69, 40, 0, 0]]],
        [A.VerticalFlip, [[20, 30, 40, 45, 0]], [[20, 69, 40, 315, 0]]],
        [A.VerticalFlip, [[20, 30, 40, 90, 0]], [[20, 69, 40, 270, 0]]],
        #
        [A.SliceFlip, [[20, 30, 40, 0, 0]], [[20, 30, 59, 0, 0]]],
        [A.SliceFlip, [[20, 30, 40, 45, 0]], [[20, 30, 59, 45, 0]]],
        [A.SliceFlip, [[20, 30, 40, 90, 0]], [[20, 30, 59, 90, 0]]],
    ],
)
def test_keypoint_transform_format_xyas(aug, keypoints, expected):
    transform = A.Compose(
        [aug(p=1)], keypoint_params={"format": "xyzas", "angle_in_degrees": True, "label_fields": ["labels"]}
    )

    image = np.ones((100, 100, 100))
    transformed = transform(image=image, keypoints=keypoints, labels=np.ones(len(keypoints)))
    assert np.allclose(expected, transformed["keypoints"])


@pytest.mark.parametrize(
    ["keypoint", "expected", "factor", "axes"],
    [
        ((20, 30, 40, math.pi / 2, 0), (20, 30, 40, math.pi / 2, 0), 0, "xy"),
        ((20, 30, 40, math.pi / 2, 0), (30, 79, 40, 0, 0), 1, "xy"),
        ((20, 30, 40, math.pi / 2, 0), (79, 69, 40, 3 * math.pi / 2, 0), 2, "xy"),
        ((20, 30, 40, math.pi / 2, 0), (69, 20, 40, math.pi, 0), 3, "xy"),
        #
        ((20, 30, 40, math.pi / 2, 0), (59, 30, 20, math.pi / 2, 0), 1, "xz"),
        ((20, 30, 40, math.pi / 2, 0), (79, 30, 59, math.pi / 2, 0), 2, "xz"),
        
    ],
)
def test_keypoint_rotate90(keypoint, expected, factor, axes):
    actual = FGeometric.keypoint_rot90(keypoint, factor, axes=axes, rows=100, cols=100, slices=100)
    assert actual == expected


@pytest.mark.parametrize(
    ["keypoint", "expected", "angle", "axes"],
    [
        ((20, 30, 40, math.pi / 2, 0), (20, 30, 40, math.pi / 2, 0), 0, "xy"),
        ((20, 30, 40, math.pi / 2, 0), (30, 79, 40, math.pi, 0), 90, "xy"),
        ((20, 30, 40, math.pi / 2, 0), (79, 69, 40, 3 * math.pi / 2, 0), 180, "xy"),
        ((20, 30, 40, math.pi / 2, 0), (69, 20, 40, 0, 0), 270, "xy"),
        #
        ((20, 30, 40, math.pi / 2, 0), (40, 30, 79, math.pi / 2, 0), 90, "xz"),
        ((20, 30, 40, math.pi / 2, 0), (79, 30, 59, math.pi / 2, 0), 180, "xz"),

        # [[20, 30, math.pi / 2, 0], [20, 30, math.pi / 2, 0], 0],
        # [[20, 30, math.pi / 2, 0], [30, 79, math.pi, 0], 90],
        # [[20, 30, math.pi / 2, 0], [79, 69, 3 * math.pi / 2, 0], 180],
        # [[20, 30, math.pi / 2, 0], [69, 20, 0, 0], 270],
        # [[0, 0, 0, 0], [99, 99, math.pi, 0], 180],
        # [[99, 99, 0, 0], [0, 0, math.pi, 0], 180],
    ],
)
def test_keypoint_rotate(keypoint, expected, angle, axes):
    actual = FGeometric.keypoint_rotate(keypoint, angle, axes, rows=100, cols=100, slices=100, crop_to_border=False)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize(
    ["keypoint", "expected", "scale"],
    [
        [[0.0, 0.0, 0.0, math.pi / 2, 1], [0.0, 0.0, 0.0, math.pi / 2, 1], 1],
        [[0.0, 0.0, 0.0, math.pi / 2, 1], [0.0, 0.0, 0.0, math.pi / 2, 2], 2],
        [[0.0, 0.0, 0.0, math.pi / 2, 1], [0.0, 0.0, 0.0, math.pi / 2, 0.5], 0.5],
    ],
)
def test_keypoint_scale(keypoint, expected, scale):
    actual = FGeometric.keypoint_scale(keypoint, scale, scale, scale)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize(
    ["keypoint", "expected", "angle", "scale", "dx", "dy"],
    [[[50, 50, 50, 0, 5], [120, 160, 50, math.pi / 2, 10], 90, 2, 0.1, 0.1]],
)
def test_keypoint_shift_scale_rotate(keypoint, expected, angle, scale, dx, dy):
    actual = FGeometric.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, dz=0, rows=100, cols=200, slices=100)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_compose_with_additional_targets():
    image = np.ones((100, 100, 100))
    keypoints = [(10, 10, 10), (50, 50, 50)]
    kp1 = [(15, 15, 15), (55, 55, 55)]
    aug = A.Compose([A.CenterCrop(50, 50, 50)], keypoint_params={"format": "xyz"}, additional_targets={"kp1": "keypoints"})
    transformed = aug(image=image, keypoints=keypoints, kp1=kp1)
    assert transformed["keypoints"] == [(25, 25, 25)]
    assert transformed["kp1"] == [(30, 30, 30)]


@pytest.mark.parametrize(
    ["angle", "expected"],
    [
        [0, 0],
        [np.pi / 2, np.pi / 2],
        [np.pi, np.pi],
        [3 * np.pi / 2, 3 * np.pi / 2],
        [2 * np.pi, 0],
        [-np.pi / 2, 3 * np.pi / 2],
        [-np.pi, np.pi],
        [-3 * np.pi / 2, np.pi / 2],
        [-2 * np.pi, 0],
    ],
)
def test_angle_to_2pi_range(angle, expected):
    assert np.isclose(angle_to_2pi_range(angle), expected)


def test_coarse_dropout():
    aug = A.Compose(
        [A.CoarseDropout(min_holes=1, max_holes=1, min_height=128, max_width=128, min_width=128, max_height=128, min_depth=128, max_depth=128, p=1)],
        keypoint_params=A.KeypointParams(format="xyz"),
    )

    result = aug(image=np.zeros((128, 128, 128)), keypoints=((10, 10, 10), (20, 30, 40)))
    assert len(result["keypoints"]) == 0


@pytest.mark.parametrize(
    ["keypoints", "expected_keypoints", "holes"],
    [
        [[(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [], [(40, 40, 40, 60, 60, 60), (70, 70, 70, 80, 80, 80), (10, 10, 10, 20, 20, 20)]],
        [[(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [], [(10, 10, 10, 20, 20, 20), (40, 40, 40, 60, 60, 60), (70, 70, 70, 80, 80, 80)]],
        [[(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [], [(40, 40, 40, 60, 60, 60), (10, 10, 10, 20, 20, 20), (70, 70, 70, 80, 80, 80)]],
        [[(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [(75, 75, 75, 0, 0)], [(40, 40, 40, 60, 60, 60), (10, 10, 10, 20, 20, 20)]],
        [[(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [(50, 50, 50, 0, 0)], [(70, 70, 70, 80, 80, 80), (10, 10, 10, 20, 20, 20)]],
        [[(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [(50, 50, 50, 0, 0), (75, 75, 75, 0, 0)], [(10, 10, 10, 20, 20, 20)]],
    ],
)
def test_coarse_dropout_remove_keypoints(keypoints, expected_keypoints, holes):
    t = A.CoarseDropout()
    result_keypoints = t.apply_to_keypoints(keypoints, holes)

    assert set(result_keypoints) == set(expected_keypoints)
