import numpy as np
import pytest

from dicaugment import (
    RescaleSlopeIntercept,
    SetPixelSpacing,
    rescale_slope_intercept,
    reset_dicom_slope_intercept,
    dicom_scale,
    transpose_dicom,
    Compose,
    RandomScale,
    NPSNoise
)


@pytest.mark.parametrize(
    ["slope", "intercept"],
    [(1,0), (2,0), (1, -1024), (2, -1024)],
)
def test_rescale_slope_intercept_dtype(slope, intercept):
    img = np.ones([50,50,50], dtype= np.uint16)
    assert rescale_slope_intercept(img, slope, intercept).dtype == np.int16


@pytest.mark.parametrize(
    ["slope", "intercept"],
    [(1,0), (2,0), (1, -1024), (2, -1024)],
)
def test_rescale_slope_intercept_image(slope, intercept):
    img = np.ones([50,50,50], dtype= np.uint16)
    expected = (np.ones([50,50,50], dtype=np.int16) * slope) + intercept

    assert np.array_equal(rescale_slope_intercept(img, slope, intercept), expected)


@pytest.mark.parametrize(
    "dicom",
    [
        {"RescaleSlope" : 1, "RescaleIntercept": 0},
        {"RescaleSlope" : 2, "RescaleIntercept": 10},
        {"RescaleSlope" : 2, "RescaleIntercept": -1024},
        {"RescaleSlope" : 1, "RescaleIntercept": -1024},
        {"RescaleSlope" : 2, "RescaleIntercept": 0},
    ],
)
def test_rescale_slope_intercept(dicom):
    img = np.ones([50,50,50], dtype= np.uint16)
    expected_img = (np.ones([50,50,50], dtype=np.int16) * dicom["RescaleSlope"]) + dicom["RescaleIntercept"]
    expected_dcm = {"RescaleSlope" : 1, "RescaleIntercept": 0}

    aug = Compose([RescaleSlopeIntercept()])

    data = aug(image = img, dicom = dicom)

    assert data["image"].dtype == np.int16
    assert np.array_equal(data["image"], expected_img)
    assert data["dicom"] == expected_dcm



@pytest.mark.parametrize(
    "dicom",
    [
        {"RescaleSlope" : 1, "RescaleIntercept": 0},
        {"RescaleSlope" : 2, "RescaleIntercept": 10},
        {"RescaleSlope" : 2, "RescaleIntercept": -1024},
        {"RescaleSlope" : 1, "RescaleIntercept": -1024},
        {"RescaleSlope" : 2, "RescaleIntercept": 0},
    ],
)
def test_reset_dicom_slope_intercept(dicom):
    expected_dcm = {"RescaleSlope" : 1, "RescaleIntercept": 0}
    assert reset_dicom_slope_intercept(dicom) == expected_dcm

@pytest.mark.parametrize(
    ["dicom", "scale_x", "scale_y", "expected"],
    [
        ({"PixelSpacing" : (0.5,0.5)}, 1.0, 1.0, {"PixelSpacing" : (0.5,0.5)} ),
        ({"PixelSpacing" : (0.5,0.5)}, 2.0, 2.0, {"PixelSpacing" : (1.0,1.0)} ),
        ({"PixelSpacing" : (1.0,1.0)}, 0.5, 0.5, {"PixelSpacing" : (0.5,0.5)} ),
    ],
)
def test_dicom_scale(dicom, scale_x, scale_y, expected):
    assert dicom_scale(dicom, scale_x, scale_y) == expected


@pytest.mark.parametrize(
    ["dicom", "expected"],
    [
        ({"PixelSpacing" : (0.5,0.5)}, {"PixelSpacing" : (0.5,0.5)} ),
        ({"PixelSpacing" : (0.5,1.0)}, {"PixelSpacing" : (1.0,0.5)} ),
        ({"PixelSpacing" : (0.5,1.0)}, {"PixelSpacing" : (1.0,0.5)} ),
        ({"PixelSpacing" : (0.7,1.0)}, {"PixelSpacing" : (1.0,0.7)} ),
        ({"PixelSpacing" : (0.2,1.0)}, {"PixelSpacing" : (1.0,0.2)} ),
    ],
)
def test_transpose_dicom(dicom, expected):
    assert transpose_dicom(dicom) == expected



@pytest.mark.parametrize(
    ["dicom", "space_x", "space_y", "expected", "shape"],
    [
        ({"PixelSpacing" : (0.5,0.5) }, 0.5, 0.5, {"PixelSpacing" : (0.5,0.5) }, (10,10,10)),
        ({"PixelSpacing" : (0.5,0.5) }, 1.0, 1.0, {"PixelSpacing" : (1.0,1.0) }, (20,20,10)),
        ({"PixelSpacing" : (1.0,1.0) }, 0.5, 0.5, {"PixelSpacing" : (0.5,0.5) }, ( 5, 5,10)),
        ({"PixelSpacing" : (0.5,0.5) }, 0.5, 0.5, {"PixelSpacing" : (0.5,0.5) }, (10,10,10)),
        ({"PixelSpacing" : (0.5,0.5) }, 1.0, 1.0, {"PixelSpacing" : (1.0,1.0) }, (20,20,10)),
        ({"PixelSpacing" : (1.0,1.0) }, 0.5, 0.5, {"PixelSpacing" : (0.5,0.5) }, ( 5, 5,10)),
    ],
)
def test_set_pixel_spacing(dicom, space_x, space_y, expected, shape):
    img = np.ones([10,10,10], dtype= np.uint16)

    aug = Compose([SetPixelSpacing(space_x, space_y)])

    data = aug(image = img, dicom = dicom)

    assert data["image"].shape == shape
    assert data["dicom"] == expected



@pytest.mark.parametrize(
    ["dicom", "scale_limit", "expected", "shape"],
    [
        ({"PixelSpacing" : (0.5,0.5) }, ( 0.0,  0.0),  {"PixelSpacing" : (0.5,0.5) }, (10,10,10)),
        ({"PixelSpacing" : (0.5,0.5) }, ( 1.0,  1.0),  {"PixelSpacing" : (1.0,1.0) }, (20,20,20)),
        ({"PixelSpacing" : (1.0,1.0) }, (-0.5, -0.5),  {"PixelSpacing" : (0.5,0.5) }, ( 5, 5, 5)),
    ],
)
def test_set_pixel_spacing_with_random_scale(dicom, scale_limit, expected, shape):
    img = np.ones([10,10,10], dtype= np.uint16)

    aug = Compose([RandomScale(scale_limit = scale_limit, p = 1.0)])

    data = aug(image = img, dicom = dicom)

    assert data["image"].shape == shape
    assert data["dicom"] == expected


def test_nps_noise():
    img = np.ones([100,100,100], dtype= np.int16)

    dicom = {
        "PixelSpacing": (0.5, 0.5),
        "ConvolutionKernel" : 'STANDARD',
        "XRayTubeCurrent" : 160
    }

    aug = Compose([NPSNoise(p = 1.0)])
    out = aug(image = img, dicom = dicom)

    aug = Compose([NPSNoise(sample_tube_current=True, p = 1.0)])
    out = aug(image = img, dicom = dicom)

