Getting Started
=============================

Welcome to DICaugment! This guide will walk you through the process of getting started with DICaugment, a Python package for 3D image augmentation tailored specifically for volumetric 3D images such as CT scans.


Basic Usage
----------------------------------------

To use DICaugment in your Python code, follow these steps:

1. Import the necessary modules:

   .. code-block:: python
      
      import dicaugment as dca

2. Define an augmentation pipeline using `A.Compose`:

   .. code-block:: python

      transform = dca.Compose([
         # Add your desired augmentation techniques here
         # For example:
         dca.Rotate(p=0.5, limit=20, interpolation=1),
         dca.RandomCrop(p=0.5, height=64, width=64, depth=64)
      ])


3. Apply the transformation to your 3D image data:

   .. code-block:: python

      augmented_image = transform(image=scan)["image"]


   Make sure to replace `scan` with your actual 3D image data. DICaugment provides a variety of augmentation techniques specifically designed for 3D images. Some of the available techniques include:

   - `Rotate`: Rotates the 3D image along the specified axes.
   - `RandomCrop`: Randomly crops a 3D region from the input.

Next Steps
----------------------------

Congratulations! You've completed the basic setup and usage of DICaugment. Now you can explore more advanced augmentation techniques, customize the pipeline to your specific needs, and integrate DICaugment into your machine learning pipeline for 3D image analysis.

To learn more about the available classes and methods, please refer to the :doc:`API Reference <dicaugment.augmentations>`.

- If you encounter any issues during installation, please seek help from the DICaugment community on the `DICaugment GitHub Discussions <https://github.com/jjmcintosh/dicaugment/discussions>`_ page.

Happy augmenting with DICaugment!