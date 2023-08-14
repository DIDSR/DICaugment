Using Masks for Segmentation Tasks
==============================================

DICaugment not only supports augmenting 3D images but also provides functionality to augment corresponding segmentation masks with the same set of augmentations.

This guide will walk you through the process of using DICaugment to augment 3D image masks in sync with the image transformations.


1. Import the necessary modules:

    .. code-block:: python
    
        import dicaugment as dca
        import nibabel as nib


2. Read in scan and mask from disk

    .. code-block:: python
      
        scan = nib.load("path/to/scan.nii.gz").get_fdata()

    
    You can read in the mask just as you would read in a scan. The height, width, and depth of the mask must match the respective height, width, and depth of the image.

    .. code-block:: python

        mask = nib.load("path/to/mask.nii.gz").get_fdata()

    For instance segmentation, multiples masks may be needed. The resulting masks should be wrapped in a list
    
    .. code-block:: python

        mask1 = nib.load("path/to/mask1.nii.gz").get_fdata()
        mask2 = nib.load("path/to/mask2.nii.gz").get_fdata()
        mask3 = nib.load("path/to/mask3.nii.gz").get_fdata()
        masks = [mask1, mask2, mask3]


3. Define an augmentation pipeline using ``A.Compose``:

    .. code-block:: python

        transform = dca.Compose([
            # Add your desired augmentation techniques here
            # For example:
            dca.Rotate(p=0.5, limit=20, interpolation=1),
            dca.RandomCrop(height=64, width=64, depth=64)
        ])


4. Apply the transformation to your 3D image data and mask

    With a single mask and scan passing through the pipeline, ``transform`` must be called using the explicit keyword arguements: ``image`` and ``mask``, where the scan should be passed in ``image`` and the mask should be passed in ``mask``. The output of this transformation will be a dictionary that contains the augmented scan under the key ``image`` and augmented mask under the key ``mask``

    .. code-block:: python

        transformed_output = transform(image=scan, mask=mask)
        transformed_scan = transformed_output["image"]
        transformed_mask = transformed_output["mask"]


    .. image:: /_static/SegmentationExample.png
        :width: 1200px

    
    If there is more than one mask that are associated with a single scan, you should use the ``masks`` argument instead of ``mask`` where ``masks`` is a list of of individual masks.

    .. code-block:: python

        transformed_output = transform(image=scan, masks=masks)
        transformed_image = transformed_output['image']
        transformed_masks = transformed_output['masks']


    
You have learned how to use DICaugment to augment 3D image masks for segmentation tasks. Feel free to explore the wide range of augmentation techniques available in DICaugment to further enhance your segmentation tasks. For a comprehensive list of available techniques and their parameters, please refer to the :doc:`API Reference <dicaugment.augmentations>`. If you encounter any issues or have questions, please seek help from the dicaugment community on the `DICaugment GitHub Discussions <https://github.com/jjmcintosh/dicaugment/discussions>`_ page.

Happy augmenting with DICaugment in your object detection pipelines!