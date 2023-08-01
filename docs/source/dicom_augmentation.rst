Using DICOM Header Metadata in Transforms
=================================================

Albumentations3D also supports 3D image augmentations that utilize metadata from a DICOM header file to apply both pixel-level and spatial-level transformations

This guide will walk you through the process of using Albumentations3D to augment 3D images using metadata from a dicom header.


1. Import the necessary modules:

    .. code-block:: python
    
        import albumentations3d as A

2. Read in scan and dicom header from disk

    .. code-block:: python
      
        scan, dicom = A.read_dcm_image(
            path='path/to/dcm/folder/',
            return_header=True         # Set as True to recieve scan and dicom header
        )

    
    Alternatively, you may define the dicom header manually with a dictionary

    .. code-block:: python

        dicom = {
            "PixelSpacing" : (0.48, 0.48),
            "RescaleIntercept" : -1024.0,
            "RescaleSlope" : 1.0,
            "ConvolutionKernel" : 'b30f',
            "XRayTubeCurrent" : 240
        }


3. Define an augmentation pipeline using ``A.Compose``:

    .. code-block:: python

        aug = A.Compose(
            [
                # The Rescale Slope Intercept transformation converts the pixel values of a scan into Hounsfield Units (HU)
                # by using the `RescaleSlope` and `RescaleIntercept` values from the dicom header
                A.RescaleSlopeIntercept(),

                # The Set Pixel Spacing transformation resizes each slice of the scan so that the `PixelSpacing` value
                # in the dicom header is equal to `(space_x, space_y)`
                A.SetPixelSpacing(space_x = 0.5, space_y = 0.5),

                # The NPSNoise transormation applies a random change in the magnitude of the noise present in the 
                # image consistent with the kernel type provided in the DICOM header.
                A.NPSNoise(sample_tube_current = True),
                # with sample_tube_current = True, the magnitude of the noise will be
                # randomly selected from the range of [0, 500 - `XRayTubeCurrent`]
            ]
        )

4. Apply the transformation to your 3D image data and mask

    As with other augmentation pipelines, ``transform`` must be called using explicit keyword arguements: ``image`` and ``dicom``, where the scan should be passed in ``image`` and the dicom header should be passed in ``dicom``. The output of this transformation will be a dictionary that contains the augmented scan under the key ``image``.

    .. code-block:: python

        transformed_output = transform(image=scan, dicom=dicom)
        transformed_scan = transformed_output["image"]



    
