DICaugment documentation
============================================

Welcome to the documentation for `DICaugment <https://github.com/jjmcintosh/dicaugment>`_ - a Python package for 3D image augmentation, specifically designed for working with volumetric 3D images like CT scans. This guide will help you get started with dicaugment and provide comprehensive information on its features, installation, usage, and more.


What is DICaugment?
============================================
DICaugment is an extension of the popular image augmentation library Albumentations, but with additional enhancements for 3D images. It provides a collection of powerful and efficient augmentation techniques that can be seamlessly integrated into your machine learning pipeline to enhance the performance and robustness of your 3D image models.


Table of Contents
============================================
- :doc:`installation`.
   Learn how to install DICaugment and its dependencies.
- :doc:`getting.started`
   A step-by-step guide to help you quickly get started with DIcaugment.
- :doc:`API Reference <dicaugment.augmentations>`
   Detailed documentation of all available classes and methods in DICaugment.

Index
==================

* :ref:`genindex`



.. Hidden TOCs

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Getting Started:

   installation
   getting.started
   mask_augmentation
   bbox_augmentation
   dicom_augmentation
   

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference:

   dicaugment.augmentations
   dicaugment.core
   dicaugment.pytorch
   dicaugment.tensorflow

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: License:

