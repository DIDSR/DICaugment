---
title: 'Albumentations3D: A Python Package for 3D Medical Imaging Augmentation'
tags:
  - Python
  - Augmentation
  - Deep Learning
  - Medical
  - 
authors:
  - name: J. McIntosh
    orcid: 0009-0008-2573-180X
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: M. Mehdi Farhangi
    orcid: 0000-0000-0000-0000
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    corresponding: true
    affiliation: 1
affiliations:
 - name: Division of Imaging, Diagnostics, and Software Reliability, CDRH, U.S. Food and Drug Administration, Silver Spring, MD 20993, USA
   index: 1
 - name: Oak Ridge Institute for Science and Education, Oak Ridge, TN, USA
   index: 2
date: 8 August 2023
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Computer-aided diagnosis/detection (CADx/CADe) has been a prominent area of research for the subfield of medical image processing. While recent advancements in this field have increased model performance, the success of these models still largely relies on a significant amount of labeled and annotated training data. Because of the medical nature of the subject material, only medical professionals are allowed to annotate and label this medical data [CITE], which makes the annotation process extremely expensive. As a result, data augmentation plays a crucial role in increasing the size and diversity of the limited data, which in turn, reduces the possibility of overfitting and improves the capability of these models to generalize.

# Statement of need

Albumentations3D is a Python package based on the popular image augmentation library Albumentations [@info11020125] but with specific enhancements for working with volumetric 3D images, such as CT scans and other volumetric imaging. This package provides a collection of powerful and efficient augmentation techniques that can be seamlessly integrated into a machine-learning pipeline to augment 3D images and volumes.

While many image augmentation libraries, like the popular Albumentations library, excel at performing augmentations on traditional 2D or RGB images, they lack the utility to complete these augmentations on 3D images. Albumentations3D addresses this need by extending the success of the Albumentations library for 2D image augmentation to the realm of volumetric 3D images, offering a comprehensive set of transformations and augmentations, ranging from pixel-level intensity transformations to spatial transformations, all designed and optimized for 3D data. 

One aspect that sets this package apart from other 3D augmentation packages such as Volumentations [@solovyev20223d] is the inclusion of physics-based transformations that can utilize metadata commonly found in DICOM files. One example of these transformations computes and inserts into scans a random noise texture that is dependent on the convolution kernel used by the scan's manufacturer during reconstruction. [@Solomon2012-lj]. This transformation can improve data diversity and reduce unseen bias induced by manuifacturer reconstruction algorithms that are invisible to the human eye.

TALK ABOUT CURRENT RESEARCH USING NPS TRANSFORMATIONS

The Albumentations3D package provides researchers and clinicians with an efficient and user-friendly solution for augmenting 3D medical imaging datasets. Its capabilities are invaluable where data augmentation is a critical step in the training of a robust and accurate deep-learning model. By providing a comprehensive and user-friendly API for the augmentation of 3D images, Albumentations3D contributes to advancing research in fields such as medical diagnosis, object recognition, and remote sensing. In all, The Albumentations3D package aims to fill a significant gap between the computer vision community and medical imaging community by providing a specialized toolset for augmenting volumetric 3D images that follows a familiar structure commonly used by computer vision experts.



<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- # Figures -->

<!-- Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

<!-- We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project. -->

# References