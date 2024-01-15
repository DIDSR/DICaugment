import numpy as np
import pytest

from dicaugment import Crop, RandomCrop, RandomSizedCrop, Rotate
from dicaugment.core.bbox_utils import (
    calculate_bbox_area_volume,
    convert_bbox_from_dicaugment,
    convert_bbox_to_dicaugment,
    convert_bboxes_to_dicaugment,
    denormalize_bbox,
    denormalize_bboxes,
    normalize_bbox,
    normalize_bboxes,
)
from dicaugment.core.composition import BboxParams, Compose, ReplayCompose
from dicaugment.core.transforms_interface import NoOp


@pytest.mark.parametrize(
    ["bbox", "expected"],
    [
        ((15, 25, 30, 100, 200, 450), (0.0375, 0.125, 0.05, 0.25, 1.0, 0.75)),
        ((15, 25, 30, 100, 200, 450, 99), (0.0375, 0.125, 0.05, 0.25, 1.0, 0.75, 99)),
    ],
)
def test_normalize_bbox(bbox, expected):
    normalized_bbox = normalize_bbox(bbox, 200, 400, 600)
    assert normalized_bbox == expected


@pytest.mark.parametrize(
    ["bbox", "expected"],
    [
        ((0.0375, 0.125, 0.05, 0.25, 1.0, 0.75), (15, 25, 30, 100, 200, 450)),
        ((0.0375, 0.125, 0.05, 0.25, 1.0, 0.75, 99), (15, 25, 30, 100, 200, 450, 99)),
    ],
)
def test_denormalize_bbox(bbox, expected):
    denormalized_bbox = denormalize_bbox(bbox, 200, 400, 600)
    assert denormalized_bbox == expected


@pytest.mark.parametrize(
    "bbox", [(15, 25, 30, 100, 200, 150), (15, 25, 30, 100, 200, 150, 99)]
)
def test_normalize_denormalize_bbox(bbox):
    normalized_bbox = normalize_bbox(bbox, 200, 400, 600)
    denormalized_bbox = denormalize_bbox(normalized_bbox, 200, 400, 600)
    assert denormalized_bbox == bbox


@pytest.mark.parametrize(
    "bbox",
    [
        (0.0375, 0.125, 0.05, 0.25, 1.0, 0.75),
        (0.0375, 0.125, 0.05, 0.25, 1.0, 0.75, 99),
    ],
)
def test_denormalize_normalize_bbox(bbox):
    denormalized_bbox = denormalize_bbox(bbox, 200, 400, 600)
    normalized_bbox = normalize_bbox(denormalized_bbox, 200, 400, 600)
    assert normalized_bbox == bbox


def test_normalize_bboxes():
    bboxes = [(15, 25, 30, 100, 200, 450), (15, 25, 30, 100, 200, 450, 99)]
    normalized_bboxes_1 = normalize_bboxes(bboxes, 200, 400, 600)
    normalized_bboxes_2 = [
        normalize_bbox(bboxes[0], 200, 400, 600),
        normalize_bbox(bboxes[1], 200, 400, 600),
    ]
    assert normalized_bboxes_1 == normalized_bboxes_2


def test_denormalize_bboxes():
    bboxes = [
        (0.0375, 0.125, 0.05, 0.25, 1.0, 0.75),
        (0.0375, 0.125, 0.05, 0.25, 1.0, 0.75, 99),
    ]
    denormalized_bboxes_1 = denormalize_bboxes(bboxes, 200, 400, 600)
    denormalized_bboxes_2 = [
        denormalize_bbox(bboxes[0], 200, 400, 600),
        denormalize_bbox(bboxes[1], 200, 400, 600),
    ]
    assert denormalized_bboxes_1 == denormalized_bboxes_2


@pytest.mark.parametrize(
    ["bbox", "rows", "cols", "slices", "expected"],
    [
        ((0, 0, 0, 1, 1, 1), 50, 100, 10, (5000, 50000)),
        ((0.2, 0.2, 0.2, 1, 1, 1, 99), 10, 10, 10, (64, 512)),
    ],
)
def test_calculate_bbox_area_volume(bbox, rows, cols, slices, expected):
    area_vol = calculate_bbox_area_volume(bbox, rows, cols, slices)
    assert area_vol == expected


@pytest.mark.parametrize(
    ["bbox", "source_format", "expected"],
    [
        ((20, 30, 40, 40, 50, 50), "coco_3d", (0.2, 0.3, 0.4, 0.6, 0.8, 0.9)),
        ((20, 30, 40, 40, 50, 50, 99), "coco_3d", (0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 99)),
        ((20, 30, 40, 60, 80, 90), "pascal_voc_3d", (0.2, 0.3, 0.4, 0.6, 0.8, 0.9)),
        (
            (20, 30, 40, 60, 80, 90, 99),
            "pascal_voc_3d",
            (0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 99),
        ),
        ((0.2, 0.3, 0.5, 0.4, 0.5, 0.2), "yolo_3d", (0.00, 0.05, 0.4, 0.40, 0.55, 0.6)),
        (
            (0.2, 0.3, 0.5, 0.4, 0.5, 0.2, 99),
            "yolo_3d",
            (0.00, 0.05, 0.4, 0.40, 0.55, 0.6, 99),
        ),
        ((0.1, 0.1, 0.1, 0.2, 0.2, 0.2), "yolo_3d", (0.0, 0.0, 0.0, 0.2, 0.2, 0.2)),
        (
            (0.99662423, 0.7520255, 0.5383403, 0.00675154, 0.01446759, 0.0294565),
            "yolo_3d",
            (0.99324846, 0.744791705, 0.52361205, 1.0, 0.759259295, 0.55306855),
        ),
        (
            (0.9375, 0.510416, 0.543219, 0.1234375, 0.97638, 0.483940),
            "yolo_3d",
            (0.87578125, 0.022226, 0.301249, 0.999218749, 0.998606, 0.785189),
        ),
    ],
)
def test_convert_bbox_to_dicaugment(bbox, source_format, expected):
    image = np.ones((100, 100, 100))

    converted_bbox = convert_bbox_to_dicaugment(
        bbox,
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format=source_format,
    )
    assert np.all(np.isclose(converted_bbox, expected))


@pytest.mark.parametrize(
    ["bbox", "target_format", "expected"],
    [
        ((0.2, 0.3, 0.4, 0.6, 0.8, 0.9), "coco_3d", (20, 30, 40, 40, 50, 50)),
        ((0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 99), "coco_3d", (20, 30, 40, 40, 50, 50, 99)),
        ((0.2, 0.3, 0.4, 0.6, 0.8, 0.9), "pascal_voc_3d", (20, 30, 40, 60, 80, 90)),
        (
            (0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 99),
            "pascal_voc_3d",
            (20, 30, 40, 60, 80, 90, 99),
        ),
        (
            (0.00, 0.05, 0.3, 0.40, 0.55, 0.6),
            "yolo_3d",
            (0.2, 0.3, 0.45, 0.4, 0.5, 0.3),
        ),
        (
            (0.00, 0.05, 0.3, 0.40, 0.55, 0.6, 99),
            "yolo_3d",
            (0.2, 0.3, 0.45, 0.4, 0.5, 0.3, 99),
        ),
    ],
)
def test_convert_bbox_from_dicaugment(bbox, target_format, expected):
    image = np.ones((100, 100, 100))
    converted_bbox = convert_bbox_from_dicaugment(
        bbox,
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        target_format=target_format,
    )
    assert np.all(np.isclose(converted_bbox, expected))


@pytest.mark.parametrize(
    ["bbox", "bbox_format"],
    [
        ((20, 30, 40, 40, 50, 50), "coco_3d"),
        ((20, 30, 40, 40, 50, 50, 99), "coco_3d"),
        ((20, 30, 40, 41, 51, 51, 99), "coco_3d"),
        ((21, 31, 41, 40, 50, 50, 99), "coco_3d"),
        ((21, 31, 41, 41, 51, 51, 99), "coco_3d"),
        ((20, 30, 40, 60, 80, 90), "pascal_voc_3d"),
        ((20, 30, 40, 60, 80, 90, 99), "pascal_voc_3d"),
        ((20, 30, 40, 61, 81, 91, 99), "pascal_voc_3d"),
        ((21, 31, 41, 60, 80, 90, 99), "pascal_voc_3d"),
        ((21, 31, 41, 61, 81, 91, 99), "pascal_voc_3d"),
        ((0.25, 0.3, 0.5, 0.4, 0.5, 0.2), "yolo_3d"),
        ((0.25, 0.3, 0.5, 0.4, 0.5, 0.2, 99), "yolo_3d"),
        ((0.25, 0.3, 0.5, 0.41, 0.51, 0.21, 99), "yolo_3d"),
        ((0.26, 0.31, 0.51, 0.4, 0.5, 0.2, 99), "yolo_3d"),
        ((0.26, 0.31, 0.51, 0.41, 0.51, 0.21, 99), "yolo_3d"),
    ],
)
def test_convert_bbox_to_dicaugment_and_back(bbox, bbox_format):
    image = np.ones((100, 100, 100))
    converted_bbox = convert_bbox_to_dicaugment(
        bbox,
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format=bbox_format,
    )
    converted_back_bbox = convert_bbox_from_dicaugment(
        converted_bbox,
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        target_format=bbox_format,
    )
    assert np.all(np.isclose(converted_back_bbox, bbox))


def test_convert_bboxes_to_dicaugment():
    bboxes = [(20, 30, 40, 40, 50, 50), (10, 20, 30, 40, 50, 60, 99)]
    image = np.ones((100, 100, 100))
    converted_bboxes = convert_bboxes_to_dicaugment(
        bboxes,
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format="coco_3d",
    )
    converted_bbox_1 = convert_bbox_to_dicaugment(
        bboxes[0],
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format="coco_3d",
    )
    converted_bbox_2 = convert_bbox_to_dicaugment(
        bboxes[1],
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format="coco_3d",
    )
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


def test_convert_bboxes_from_dicaugment():
    bboxes = [(0.2, 0.3, 0.4, 0.6, 0.8, 0.9), (0.3, 0.4, 0.5, 0.7, 0.9, 0.9, 99)]
    image = np.ones((100, 100, 100))
    converted_bboxes = convert_bboxes_to_dicaugment(
        bboxes,
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format="coco_3d",
    )
    converted_bbox_1 = convert_bbox_to_dicaugment(
        bboxes[0],
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format="coco_3d",
    )
    converted_bbox_2 = convert_bbox_to_dicaugment(
        bboxes[1],
        rows=image.shape[0],
        cols=image.shape[1],
        slices=image.shape[2],
        source_format="coco_3d",
    )
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


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
def test_compose_with_bbox_noop(bboxes, bbox_format, labels):
    image = np.ones((100, 100, 100))
    if labels is not None:
        aug = Compose(
            [NoOp(p=1.0)],
            bbox_params={"format": bbox_format, "label_fields": ["labels"]},
        )
        transformed = aug(image=image, bboxes=bboxes, labels=labels)
    else:
        aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format})
        transformed = aug(image=image, bboxes=bboxes)
    assert np.array_equal(transformed["image"], image)
    assert np.all(np.isclose(transformed["bboxes"], bboxes))


@pytest.mark.parametrize(
    ["bboxes", "bbox_format"], [[[[20, 30, 40, 40, 50, 50]], "coco_3d"]]
)
def test_compose_with_bbox_noop_error_label_fields(bboxes, bbox_format):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format})
    with pytest.raises(Exception):
        aug(image=image, bboxes=bboxes)


@pytest.mark.parametrize(
    ["bboxes", "bbox_format", "labels"],
    [
        [[(20, 30, 40, 60, 80, 90)], "pascal_voc_3d", {"label": [1]}],
        [[], "pascal_voc_3d", {}],
        [[], "pascal_voc_3d", {"label": []}],
        [[(20, 30, 40, 60, 80, 90)], "pascal_voc_3d", {"id": [3]}],
        [
            [(20, 30, 40, 60, 80, 90), (30, 40, 40, 40, 50, 60)],
            "pascal_voc_3d",
            {"id": [3, 1]},
        ],
        [
            [(20, 30, 60, 80, 40, 70), (30, 40, 40, 40, 50, 60, 2, 22)],
            "pascal_voc_3d",
            {"id": [3, 1]},
        ],
        [
            [(20, 30, 60, 80, 40, 70), (30, 40, 40, 40, 50, 60, 2, 22)],
            "pascal_voc_3d",
            {},
        ],
        [
            [(20, 30, 60, 80, 40, 70), (30, 40, 40, 40, 50, 60, 2, 21)],
            "pascal_voc_3d",
            {"id": [31, 32], "subclass": [311, 321]},
        ],
    ],
)
def test_compose_with_bbox_noop_label_outside(bboxes, bbox_format, labels):
    image = np.ones((100, 100, 100))
    aug = Compose(
        [NoOp(p=1.0)],
        bbox_params={"format": bbox_format, "label_fields": list(labels.keys())},
    )
    transformed = aug(image=image, bboxes=bboxes, **labels)
    assert np.array_equal(transformed["image"], image)
    assert transformed["bboxes"] == bboxes
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size():
    image = np.ones((100, 100, 100))
    bboxes = [(0.2, 0.3, 0.4, 0.6, 0.8, 0.9), (0.3, 0.4, 0.5, 0.7, 0.9, 0.8, 99)]
    aug = RandomSizedCrop(min_max_height=(70, 90), height=50, width=50, depth=50, p=1.0)
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 50)
    assert len(bboxes) == len(transformed["bboxes"])

def test_random_rotate():
    image = np.ones((100, 100, 100))
    bboxes = [(0.2, 0.3, 0.4, 0.6, 0.8, 0.9)]
    aug = Rotate(limit=15, p=1.0)
    transformed = aug(image=image, bboxes=bboxes)
    assert len(bboxes) == len(transformed["bboxes"])


def test_crop_boxes_replay_compose():
    image = np.ones((512, 384, 555))
    bboxes = [
        (78, 42, 20, 95, 80, 92),
        (32, 12, 10, 42, 72, 69),
        (20, 10, 40, 30, 20, 60),
    ]
    labels = [0, 1, 2]
    transform = ReplayCompose(
        [RandomCrop(64, 64, 64, p=1.0)],
        bbox_params=BboxParams(
            format="pascal_voc_3d", min_planar_area=16, label_fields=["labels"]
        ),
    )

    input_data = dict(image=image, bboxes=bboxes, labels=labels)
    transformed = transform(**input_data)
    transformed2 = ReplayCompose.replay(transformed["replay"], **input_data)

    np.testing.assert_almost_equal(transformed["bboxes"], transformed2["bboxes"])


@pytest.mark.parametrize(
    [
        "transforms",
        "bboxes",
        "result_bboxes",
        "min_planar_area",
        "min_volume_visibility",
    ],
    [
        [[Crop(10, 10, 10, 20, 20, 20)], [[0, 0, 0, 10, 10, 10, 0]], [], 0, 0],
        [
            [Crop(0, 0, 0, 90, 90, 90)],
            [[0, 0, 0, 91, 91, 91, 0], [0, 0, 0, 90, 90, 90, 0]],
            [[0, 0, 0, 90, 90, 90, 0]],
            0,
            1,
        ],
        [
            [Crop(0, 0, 0, 90, 90, 90)],
            [[0, 0, 0, 1, 10, 10, 0], [0, 0, 0, 1, 11, 11, 0]],
            [[0, 0, 0, 1, 10, 10, 0], [0, 0, 0, 1, 11, 11, 0]],
            10,
            0,
        ],
    ],
)
def test_bbox_params_edges(
    transforms, bboxes, result_bboxes, min_planar_area, min_volume_visibility
):
    image = np.empty([100, 100, 100], dtype=np.int16)
    aug = Compose(
        transforms,
        bbox_params=BboxParams(
            "pascal_voc_3d",
            min_planar_area=min_planar_area,
            min_volume_visibility=min_volume_visibility,
        ),
    )
    res = aug(image=image, bboxes=bboxes)["bboxes"]
    assert np.allclose(res, result_bboxes)
