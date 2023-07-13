Getting Started
=============================

Welcome to Albumentations3D! This guide will walk you through the process of getting started with Albumentations3D, a Python package for 3D image augmentation tailored specifically for volumetric 3D images such as CT scans.


Basic Usage
----------------------------------------

To use Albumentations3D in your Python code, follow these steps:

1. Import the necessary modules:

   .. code-block:: python
      
      import albumentations3d as A

2. Define an augmentation pipeline using `A.Compose`:

   .. code-block:: python

      transform = A.Compose([
         # Add your desired augmentation techniques here
         # For example:
         A.Rotate(p=0.5, limit=20, interpolation=1),
         A.RandomCrop(p=0.5, height=64, width=64, depth=64)
      ])


3. Apply the transformation to your 3D image data:

   .. code-block:: python

      augmented_image = transform(image=scan)["image"]


   Make sure to replace `scan` with your actual 3D image data. Albumentations3D provides a variety of augmentation techniques specifically designed for 3D images. Some of the available techniques include:

   - `Rotate`: Rotates the 3D image along the specified axes.
   - `RandomCrop`: Randomly crops a 3D region from the input.

Next Steps
----------------------------

Congratulations! You've completed the basic setup and usage of Albumentations3D. Now you can explore more advanced augmentation techniques, customize the pipeline to your specific needs, and integrate Albumentations3D into your machine learning pipeline for 3D image analysis.

To learn more about the available classes and methods, please refer to the :doc:`API Reference <albumentations3d.augmentations>`.

- If you encounter any issues during installation, please seek help from the Albumentations3D community on the `Albumentations3D GitHub Discussions <https://github.com/jjmcintosh/albumentations3d/discussions>`_ page.

Happy augmenting with Albumentations3D!