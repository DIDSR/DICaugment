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
    affiliation: 1
    corresponding: true
  - name: Qian Cao
    equal-contrib: false
    affiliation: 1
  - name: Berkman Sahiner
    equal-contrib: false
    affiliation: 1
  - name: Nicholas Petrick
    equal-contrib: false
    affiliation: 1
  - name: M. Mehdi Farhangi
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Division of Imaging, Diagnostics, and Software Reliability, CDRH, U.S. Food and Drug Administration, Silver Spring, MD 20993, USA
   index: 1
date: 8 August 2023
bibliography: paper.bib

---

# Summary

Albumentations3D is a Python package based on the popular image augmentation library Albumentations [@info11020125], with specific enhancements for working with volumetric 3D medical images, such as CT scans. This package provides a collection of powerful and efficient augmentation techniques that can be seamlessly integrated into a machine-learning pipeline to augment volumetric medical images. The package was designed to incorporate the metadata available in DICOM headers, allowing users to create transformations that are consistent with an image's acquisition parameters for specific CT systems and reconstruction kernels. Albumentations3D extends the success of the Albumentations library for two-dimensional (2D) image augmentation to the realm of three-dimensional (3D) images, offering a comprehensive set of transformations and augmentations, ranging from pixel-level intensity transformations to spatial transformations, all designed and optimized for 3D data.

# Statement of need

The recent advancements in machine learning have significantly improved the performance of deep learning models across various domains and applications. However, the success of these models still largely rely on a significant amount of labeled and annotated training data. This is a particular limitation in medical imaging applications where imaging data annotation and the establishment of a reliable reference standard is expensive, time consuming, and typically requires clinician expertise. Albumentations3D supports the efficient use of available labeled and annotated data by providing augmentations that are consistent with preexisting imaging data annotations available in the form of masks, bounding boxes, and keypoints. 

Albumentations3D provides a variety of transformations designed to enhance the diversity of 3D image data. A subset of these transformations includes blurring techniques, such as median blur and Gaussian blur, that must utilize convolution to achieve the desired output. Notably, Albumentations3D offers a unique capability by allowing users to choose between a 2D by-slice convolution or a full 3D convolution when applying these blurring augmentations in a pipeline. This flexibility caters to a variety of use cases and ensures that the blurring effects can be consistent across an entire 3D volume, preserving the spatial relationships between slices. 

Additionally, Albumentations3D enables users to construct custom augmentation pipelines with control over each individual transformation. Mimicking the Albumentations [@info11020125] package, Albumentations3D utilizes the concept of augmentation probabilities. This feature allows users to assign a probability of occurrence to each transformation within a pipeline, enabling selective and stochastic augmentation of an image dataset. By specifying different probabilities for various transformations, users can create diverse and balanced augmented datasets that reflect real-world variability. This level of control ensures that specific augmentations, whether it be intensity adjustments or geometric transformations, are applied with user-defined frequencies, therefore tailoring the augmentation process to the characteristics and use case of the dataset.

Besides different incorporations for volumetric images, Albumentation3D offers a unique advantage through the addition of physics-based augmentations that leverage metadata from DICOM files in CT imaging. During the reconstruction process of CT images, different manufacturers employ different reconstruction kernels, leading to distinct noise textures in the resulting images often characterized by the Noise Power Spectrum [@Solomon2012-lj]. Albumentation3D utilizes the NPS profiles from different reconstruction kernels to generate noisy images that are consistent with the imaging acquisition and reconstruction parameters [@Tward2008-gz]. To illustrate, we computed the Noise Power Spectrum (NPS) obtained from the noise insertion transformations offered by Albumentations3D and compared it against simple white Gaussian noise, which is a common noise insertion method for data augmentation (e.g., MONAI [@cardoso2022monai] and Volumentations [@solovyev20223d]). 

For this experiment, we utilized a water phantom [@Phantom_testing] scanned by SIEMENS with an exposure of 240 mAs, 120 kVp, and reconstructed with a B30f reconstruction kernel at 0.48 mm in-plane pixel spacing. The 2D NPS was estimated using the method proposed by Solomon et al. [@Solomon2012-lj]; from the water phantom, we extracted a uniform volumetric region of 128 $\times$ 128 $\times$ 70 dimensions and computed the 2D NPS of each slice by taking the square of the Fourier transform:


\begin{equation}\label{eq:2dnps}
NPS(u,v) = \frac{d_xd_y}{N_xN_y}.\mid F[I(x,y)-P(x,y)]\mid^2
\end{equation}


In \autoref{eq:2dnps}, $u$ and $v$ represent spatial frequency ($mm^{-1}$) in the $x$ and $y$ directions, respectively, $d_x$ and $d_y$ are pixel size in millimeter space, $N_x$ and $N_y$ are the number of pixels in the $x$ and $y$ directions within the selected ROI (i.e. 128 $\times$ 128 px), $F$ denotes the 2D Fourier transform, $I(x,y)$ is the pixel density of the uniform region in Hounsfield Unit at position $(x,y)$ and $P$ is the second order polynomial fit of $I$. Each estimated 2D NPS within the volumetric ROI is normalized by its integral across all frequencies and converted into a one-dimensional radial representation, $r^2 = \sqrt{u^2+v^2}$. The final normalized NPS (nNPS) was obtained by averaging the radial NPS curves across 70 slices within the volumetric image.  

![(a) Water phantom scanned by SIEMENS with imaging parameters set at 120 kVp, 0.48 mm pixel spacing, and an exposure of 240 mAs, and reconstructed using the B30f kernel. The noise magnitude is computed as 9.46. (b) Increasing the noise magnitude by adding white Gaussian noise, resulting in a noisy image with a magnitude of 16.01. (c) Increasing the noise magnitude using Albumentations3D, resulting in a noise image with a magnitude of 16.08.  Figures (d), (e), and (f) illustrate the benefits of Albumentations3D in offering augmentations consistent with the imaging parameters, demonstrated in terms of the Noise Power Spectrum (NPS) of the augmented image.\label{fig:example}](fig_1.png)

Figure 1(a) illustrates a slice of the noise region extracted from the uniform water phantom along with the corresponding estimation of the normalized Noise Power Spectrum (nNPS). Figure 2(b) illustrates the outcome of noise insertion using white Gaussian noise, a common approach for noise insertion [@cardoso2022monai], demonstrating the lack of correlation between adjacent pixels. On the right, the outcome of noise insertion by Albumentation3D is illustrated. Despite the increased magnitude of the noise, the correlation among adjacent pixels remained essentially unchanged, as evident in the normalized Noise Power Spectrum (nNPS) shown at the bottom of this figure. This shows that an increased CT exposure can be simulated through our augmentation process for specific CT imaging systems and kernels.

# Acknowledgements

Jacob McIntosh was supported by an appointment to the Research Participation Program at the Center for Devices and Radiological Health administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and the U.S. Food and Drug Administration. The mention of softwares, commercial products, their sources, or their use in connection with material reported herein is not to be construed as either an actual or implied endorsement of such products by the Department of Health and Human Services.

# References