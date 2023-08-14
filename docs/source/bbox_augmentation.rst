Using Bounding Boxes For Object Detection Tasks
=================================================


Annotatation Formats
------------------------


3D Bounding Boxes are cuboids that encapsulate an object within a volumetric image. There are many formats to annotate bounding boxes, and dicaugment supports 4 formats: ``pascal_voc_3d``, ``albumentations_3d``, ``coco_3d``, and ``yolo_3d``.

The following scan has a height of 512px, a width of 512px, and a depth of 64px. The width of the bounding box in this scan is 45px, while the height is 46px, and the depth is 20px.



    .. image:: /_static/BboxPointsExample.png
        :width: 1200px


pascal_voc_3d
~~~~~~~~~~~~~~~~
``pascal_voc_3d`` is a format derived from the Pascal VOC dataset. Coordinates of a bounding box are encoded with six values in pixels: ``[x_min, y_min, z_min, x_max, y_max, z_max]``. ``x_min``, ``y_min``, and ``z_min`` are coordinates of the forward-top-left corner of the bounding box. ``x_max``, ``y_max``, and ``z_max`` are coordinates of furthest-bottom-right corner of the bounding box.
Coordinates of the example bounding box in this format are ``[265, 211, 0, 310, 257, 20]``.


albumentations_3d
~~~~~~~~~~~~~~~~~

``albumentations_3d`` is the internal format for dicaugment that uses the same coordinates annotation formats as pascal_voc_3d but is normalized by the height, width, and depth of the image.
Coordinates of the example bounding box in this format are ``[265 / 512, 211 / 512, 0 / 64, 310 / 512, 257 / 512, 20 / 64]`` which simplifies to ``[0.5176, 0.4121, 0, 0.6055, 0.502, 0.3125]``.


coco_3d
~~~~~~~~

``coco_3d`` is a format derived from the Common Objects in Context dataset

In ``coco_3d``, a bounding box is defined by six pixel values ``[x_min, y_min, z_min, width, height, depth]``. They are coordinates of the forward-top-left corner along with the width, height, and depth of the bounding box.

Coordinates of the example bounding box in this format are ``[265, 211, 0, 45, 46, 20]``.

yolo_3d
~~~~~~~~

In ``yolo_3d``, the bounding box is represented by ``[x_center, y_center, z_center, width, height, depth]`` where ``x_center``, ``y_center``, and ``z_center`` are the normalized coordinates of the center of the bounding box and ``height``, ``width``, and ``depth`` are the normalized height, width, and depth of the bounding box.

Coordinates of the example bounding box in this format are ``[((265 + 310) / 2) / 512, ((211 + 257) / 2) / 512, ((0 + 20) / 2) / 64, 45 / 512, 46 / 512, 20 / 64]`` which simplifies to ``[0.5615, 0.4570, 0.15625, 0.0879, 0.0898, 0.3125]``.


Augmenting Bounding Boxes
---------------------------


1. Import the necessary modules:

    .. code-block:: python
    
        import dicaugment as dca


2. Define an augmentation pipeline using ``A.Compose``:

    .. code-block:: python

        transform = dca.Compose([
            dca.Rotate(p=0.5, limit=20, interpolation=1),
            dca.RandomCrop(height=64, width=64, depth=64)
            ],
            bbox_params= dca.BboxParams(format='pascal_voc_3d')
        )

    Note that unlike augmenting only images and masks, if you wish to augment bounding boxes, you must pass an instance of a ``BboxParams`` object to the ``bbox_params`` parameter in the declaration of ``A.Compose``. The ``BboxParams`` object is critical to the pipeline when augmenting bounding boxes because it specifies the annotation format of the bounding boxes that will be passed through the pipeline.

    ``format`` is a required argument for ``BboxParams`` and must be one of ``pascal_voc_3d``, ``dicaugment_3d``, ``coco_3d``, and ``yolo_3d``.

Filtering Bounding Boxes
~~~~~~~~~~~~~~~~~~~~~~~~~
    
    There are additional optional arguments for ``BboxParams`` that may be useful in filtering out bounding boxes that may not be useful after a particular transformation.

    .. code-block:: python

        transform = dca.Compose([
            dca.Rotate(p=0.5, limit=20, interpolation=1),
            dca.RandomCrop(height=64, width=64, depth=64)
            ],
            bbox_params= dca.BboxParams(format='pascal_voc_3d', min_planar_area = 400, min_volume = 4000)
        )

    ``min_planar_area`` and ``min_volume`` are some of many parameters for the ``BboxParams`` object that dictate how a pipeline should handle a bounding box if its shape has changed due to a transform such as resizing or cropping.

    ``min_planar_area`` is the minimum area of the 'xy' dimension of the bounding box that is required after a transform in order to be maintained. If the resulting area of a transformed bounding box does not satisfy this condition, then it will be removed and not be returned from the pipeline.

    ``min_volume`` is the minimum volume of the bounding box that is required after a transform in order to be maintained. If the resulting volume of a transformed bounding box does not satisfy this condition, then it will be removed and will not be returned from the pipeline.

    See more parameter options in the documentation for `BboxParams <https://dicaugment.readthedocs.io/en/latest/dicaugment.core.html#dicaugment.core.bbox_utils.BboxParams>`_

    
Class Labels for Bounding Boxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most bounding box annotations have associated classes or labels. In DICaugment, labels are required for bounding boxes. There are two primary ways to incorporate labels into the pipeline.

Internal Labels
"""""""""""""""""""

    You may include class labels within each bounding box annotation.
    
    The following example bounding box labeled annotations are valid and acceptable:

    .. code-block:: python

        bboxes = [
            [15, 25, 30, 100, 200, 450, "A"],
            [20, 30, 40,  40,  50,  50, "B"],
            [10, 12,  5,  60, 100,  25, "B"],
            [20, 30, 40,  60,  80,  90, "C"],
        ]


    .. code-block:: python

        bboxes = [
            [15, 25, 30, 100, 200, 450, 0],
            [20, 30, 40,  40,  50,  50, 1],
            [10, 12,  5,  60, 100,  25, 1],
            [20, 30, 40,  60,  80,  90, 2],
        ]

    .. code-block:: python

        bboxes = [
            [15, 25, 30, 100, 200, 450, "A", True],
            [20, 30, 40,  40,  50,  50, "B", False],
            [10, 12,  5,  60, 100,  25, "B", True],
            [20, 30, 40,  60,  80,  90, "C", False],
        ]


    .. code-block:: python

        bboxes = [
            [15, 25, 30, 100, 200, 450, "A", 2],
            [20, 30, 40,  40,  50,  50, "B", 0],
            [10, 12,  5,  60, 100,  25, "B", 1],
            [20, 30, 40,  60,  80,  90, "C", 2],
        ]


    Note that labels can be any pythonic object such as strings, integers, and booleans. Bounding box annotations are also allowed to have multiple class labels as shown above.

    With internal labels for each bounding box, the bounding box may be passed into the pipeline normally and the labels will be unnafected.

    .. code-block:: python

        transform = dca.Compose([
            dca.Rotate(p=0.5, limit=20, interpolation=1),
            dca.RandomCrop(height=64, width=64, depth=64)
            ],
            bbox_params= dca.BboxParams(format='pascal_voc_3d', min_planar_area = 400, min_volume = 4000)
        )

        transformed = transform(image = scan, bboxes = bboxes)
        transformed_scan   = transformed["image"]
        transformed_bboxes = transformed["bboxes"]


External Labels
"""""""""""""""""

    You may also include an external list of labels to be passed through the pipeline

    Given the following example bounding box annotation

    .. code-block:: python

        bboxes = [
            [15, 25, 30, 100, 200, 450],
            [20, 30, 40,  40,  50,  50],
            [10, 12,  5,  60, 100,  25],
            [20, 30, 40,  60,  80,  90],
        ]

    The following external example labels are valid and acceptable 

    .. code-block:: python

        class_labels = [ "A",   "B",  "B",   "C"]
        class_labels = [True, False, True, False]
        class_labels = [   0,     1,    1,     2]

    Note that if external labels are used, then the ``label_fields`` argument must be used in the ``BboxParams`` declaration to tell the pipeline what keyword argument/s to expect for all class labels passed through the pipeline

    .. code-block:: python

        transform = dca.Compose([
            dca.Rotate(p=0.5, limit=20, interpolation=1),
            dca.RandomCrop(height=64, width=64, depth=64)
            ],
            bbox_params= dca.BboxParams(format='pascal_voc_3d', min_planar_area = 400, min_volume = 4000, label_fields=['class_labels'])
        )

        transformed = transform(image = scan, bboxes = bboxes, class_labels = class_labels)
        transformed_scan   = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_labels = transformed["class_labels"]


    If the bounding boxes have more than one label associated with them, then you may pass multiple lists of labels through the pipeline as long as each list is given a keyword argument in the ``label_fields`` parameter in ``BboxParams``
    
    .. code-block:: python

        
        class_labels     = [True, False, True, False]
        class_categories = [ "A",   "B",  "B",   "C"]

        transform = dca.Compose([
            dca.Rotate(p=0.5, limit=20, interpolation=1),
            dca.RandomCrop(height=64, width=64, depth=64)
            ],
            bbox_params= dca.BboxParams(format='pascal_voc_3d', min_planar_area = 400, min_volume = 4000, label_fields=['class_labels', 'class_categories'])
        )

        transformed = transform(image = scan, bboxes = bboxes, class_labels = class_labels, class_categories = class_categories)
        transformed_scan   = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_class_labels = transformed["class_labels"]
        transformed_class_catergories = transformed["class_categories"]
    



.. image:: /_static/BboxTranformExample.png
        :width: 1200px

.. 1. Apply the transformation to your 3D image data and mask

..     With a single mask and scan passing through the pipeline, ``transform`` must be called using the explicit keyword arguements: ``image`` and ``mask``, where the scan should be passed in ``image`` and the mask should be passed in ``mask``. The output of this transformation will be a dictionary that contains the augmented scan under the key ``image`` and augmented mask under the key ``mask``

..     .. code-block:: python

..         transformed_output = transform(image=scan, mask=mask)
..         transformed_scan = transformed_output["image"]
..         transformed_mask = transformed_output["mask"]




    
..     If there is more than one mask that are associated with a single scan, you should use the ``masks`` argument instead of ``mask`` where ``masks`` is a list of of individual masks.

..     .. code-block:: python

..         transformed_output = transform(image=scan, masks=masks)
..         transformed_image = transformed_output['image']
..         transformed_masks = transformed_output['masks']


    
You have learned how to use dicaugment to augment 3D image bounding boxes for object detection. Feel free to explore the wide range of augmentation techniques available in dicaugment to further enhance your object detection tasks. For a comprehensive list of available techniques and their parameters, please refer to the :doc:`API Reference <dicaugment.augmentations>`. If you encounter any issues or have questions, please seek help from the dicaugment community on the `dicaugment GitHub Discussions <https://github.com/jjmcintosh/dicaugment/discussions>`_ page.

Happy augmenting with DICaugment in your object detection pipeline!