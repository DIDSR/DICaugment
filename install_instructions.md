1. activate venv (`. venv\bin\activate`)
2. If not already there, clone repository
    - `git clone https\\albumenations3D\github\path.git`
    - enter username
    - enter password (personal access token)
3. `pip install setuptools` (if not already installed)
4. `pip install wheel` (if not already installed)
5. `python setup.py bdist_wheel sdist` (could be python3 instead of python depending on distro)
6. open new venv in testing location (if not yest created)
    - `mkdir venv`
    - `python -m venv ./venv`
7. activate venv (`. venv\bin\activate`)
8. `pip install ../path/to/albumenations3D`
9. Test this sample script from unit test

```
import albumentations3d as A # this will eventually change to albumentations3D
import numpy as np

image = np.randint(0,256, (100,100,100), dtype = np.uint8)

# multiple bbox formats are supported, just have to pass them as an arugment, see BboxParams in core/bbox_utils.py for more info
bboxes = [(20, 30, 40, 40, 50, 50)]
bbox_format = "coco_3d"
labels =  [1]

# bboxes = [(0.1, 0.2, 0.3, 0.1, 0.2, 0.3)]
# bbox_format = "yolo_3d"
# labels =  [2]

# bboxes = [(20, 30, 40, 60, 80, 90)]
# bbox_format = "pascal_voc_3d"
# labels =  [2]

# multiple bboxes are accepted, labels are optional or can even be added to the end of the bbox, note how these have length 7 even though a bbox is length 6
# bboxes = [(20, 30, 40, 40, 50, 50, 99), (10, 40, 50, 30, 20, 25, 9)]
# bbox_format = "coco_3d"
# labels =  None


# if there are no labels, then don't inlcude the `label_fields` key in the construction of the augmentation pipeline
# format is required
if labels is None:
    bbox_params={"format": bbox_format}
else:
    bbox_params={"format": bbox_format, "label_fields": ["labels"]}


# a pipeline can be simple and linear or complex and have multiple levels
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


# virtually all pipelines can be serialized and deserialized which makes it easy to save the pipeline
# to a json and load it in again without having to reintialize all the hyperparameters
serialized_aug = A.to_dict(aug)
print(serialized_aug)


# all arguments in the augmentation must be named parameters
try:
    aug_data = aug(image, bboxes) if labels is None else aug(image, bboxes, labels)
except Exception as e:
    print(e)


# the resulting object is a dictionary of the given items
aug_data = aug(image=image, bboxes=bboxes) if labels is None else aug(image=image, bboxes=bboxes, labels=labels)
print(aug_data.keys())
print('aug_data["image"]', type(aug_data["image"]))
print('aug_data["bboxes"]', type(aug_data["bboxes"]))


# I recommend using VScode for further exploration if you can as there is an integrated terminal and there is documentation-on-hover for functions and classes


```