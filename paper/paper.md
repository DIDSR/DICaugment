---
title: 'DICaugment: A Python Package for 3D Medical Imaging Augmentation'
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

DICaugment is a Python package based on the popular image augmentation library Albumentations [@info11020125], with specific enhancements for working with volumetric medical images, such as Computed Tomography (CT) scans. This package provides a collection of over 40 powerful augmentation techniques that can be seamlessly integrated into a machine-learning pipeline to augment volumetric medical images. The package was designed to incorporate the image acquisition metadata available in DICOM headers, allowing users to create transformations that are consistent with those acquisition parameters for specific CT systems and reconstruction kernels. DICaugment extends the success of the Albumentations library for two-dimensional (2D) image augmentation to the realm of three-dimensional (3D) images, offering a comprehensive set of transformations and augmentations, ranging from pixel-level intensity transformations to spatial transformations, all designed and optimized for 3D data.

# Statement of need

The recent advancements in machine learning have significantly improved the performance of deep learning models across various domains and applications. However, the success of these models still largely rely on a significant amount of labeled and annotated training data. This is a particular limitation in medical imaging applications where imaging data annotation and the establishment of a reliable reference standard is expensive, time-consuming, and typically requires clinician expertise. DICaugment supports the efficient use of available labeled and annotated data by providing augmentations that are consistent with preexisting imaging data annotations available in the form of masks, bounding boxes, and keypoints. 

Considering the distinct geometric and operational factors inherent to 3D images, several toolkits have been developed to perform transformations in the 3D space domain; SimpleITK [@Yaniv2017] provides a collection of spatial transformations, including linear and deformable geometric transformations, as well as intensity-based transformations such as linear, non-linear, and histogram-based intensity adjustments. In addition to supporting various medical imaging tasks, augmentation methods in packages such as TorchIO [@Pérez-García2021] and MONAI [@cardoso2022monai] were designed to seamlessly integrate into popular deep learning frameworks such as PyTorch, providing a convenient interface for integration within deep learning pipelines. Packages such as Volumentations were uniquely developed for the purpose of augmenting 3D volumetric images, offering specialized transformations for enriching 3D imaging datasets during the training of deep neural networks.

DICaugment offers a variety of transformations including geometric and intensity transformations, spatial distortions, and physics-based augmentations designed to enhance the diversity of 3D image data. A subset of these transformations includes blurring techniques, such as median blur and Gaussian blur, that must utilize local spatial operations such as convolution to achieve the desired output. Notably, DICaugment offers a unique capability by allowing users to choose between a 2D by-slice operation or a full 3D operation when applying different blurring augmentations in a pipeline. This flexibility caters to a variety of use cases and ensures that the blurring effects can be consistent across an entire 3D volume. 

Additionally, DICaugment enables users to construct custom augmentation pipelines with control over each individual transformation. Mimicking the Albumentations [@info11020125] package, DICaugment utilizes the concept of augmentation probabilities. This feature allows users to assign a probability of occurrence to each transformation within a pipeline, enabling selective and stochastic augmentation of a dataset. By specifying different probabilities for various transformations, users can create diverse and balanced augmented datasets that reflect real-world variability. This level of control ensures that specific augmentations, whether it be intensity adjustments or geometric transformations, are applied with user-defined frequencies, therefore tailoring the augmentation process to the characteristics and use case of the dataset.

Compared to other augmentation packages, DICaugment offers a unique advantage through the addition of physics-based augmentations that leverage metadata from DICOM files in CT imaging. During the reconstruction process of CT images, different manufacturers employ different reconstruction kernels, leading to distinct noise textures in the resulting images often characterized by the Noise Power Spectrum (NPS) [@Solomon2012-lj]. DICaugment utilizes the NPS profiles from the specific reconstruction kernel to generate noisy images that are consistent with the imaging acquisition and reconstruction parameters [@Tward2008-gz] from the system and acquisition parameters used to acquire the original CT dataset. To illustrate, we computed the NPS obtained from the noise insertion transformations offered by DICaugment and compared it against simple white Gaussian noise, which is a common noise insertion method for data augmentation (e.g., MONAI [@cardoso2022monai] and Volumentations [@solovyev20223d]). 

For this experiment, we utilized a water phantom [@Phantom_testing] scanned by a Siemens Somatom Definition Flash CT scanner with an exposure of 240 mAs, 120 kVp, and reconstructed with a B30f reconstruction kernel at 0.48 mm in-plane pixel spacing. The 2D NPS was estimated using the method proposed by Solomon et al. [@Solomon2012-lj]; from the water phantom, we extracted a uniform volumetric region of 128 $\times$ 128 $\times$ 70 dimensions and computed the 2D NPS of each slice by using:


\begin{equation}\label{eq:2dnps}
NPS(u,v) = \frac{d_xd_y}{N_xN_y}.\mid F[I(x,y)-P(x,y)]\mid^2
\end{equation}


In \autoref{eq:2dnps}, $u$ and $v$ represent spatial frequency ($mm^{-1}$) in the $x$ and $y$ directions, respectively, $d_x$ and $d_y$ are pixel size in millimeter space, $N_x$ and $N_y$ are the number of pixels in the $x$ and $y$ directions within the selected ROI (i.e. 128 $\times$ 128 px), $F$ denotes the 2D Fourier transform, $I(x,y)$ is the pixel density of the uniform region in Hounsfield Unit at position $(x,y)$ and $P$ is the second order polynomial fit of $I$. Each estimated 2D NPS within the volumetric ROI is normalized by its integral across all frequencies and converted into a one-dimensional radial representation, $r = \sqrt{u^2+v^2}$. The final normalized NPS (nNPS) was obtained by averaging the radial NPS curves across 70 slices within the volumetric image.  

![(a) Water phantom scanned by a Siemens Somatom Definition Flash CT scanner with imaging parameters set at 240 mAs, 120 kVP, and reconstructed with a B30f reconstruction kernel at 0.48 mm in-plane pixel spacing. The noise magnitude is computed as 9.46 Hounsfield units (HU). (b) Increasing the noise magnitude by adding white Gaussian noise, resulting in a noisy image with a magnitude of 16.01 HU. (c) Increasing the noise magnitude using DICaugment, resulting in a noise image with a magnitude of 16.08 HU.  Figures (d), (e), and (f) illustrate the benefits of DICaugment in offering augmentations consistent with the imaging parameters, demonstrated in terms of the Noise Power Spectrum (NPS) of the augmented image (i.e., inserting Gaussian noise produces a high-frequency tail in the NPS that is inconsistent with that of the Siemens scanner while our noise insertion method yields a shape in the NPS that is consistent with that of the Siemens scanner).\label{fig:example}](fig_1.png)

Figure 1(a) illustrates a slice of the noise region extracted from the uniform water phantom along with the corresponding estimation of the normalized Noise Power Spectrum (nNPS). Figure 1(b) illustrates the outcome of noise insertion using white Gaussian noise, a common approach for noise insertion [@cardoso2022monai], demonstrating the lack of correlation between adjacent pixels. Figure 1(c) shows the outcome of noise insertion by DICaugment. Despite the increased magnitude of the noise, the correlation among adjacent pixels remained essentially unchanged, as evident in the normalized Noise Power Spectrum (nNPS) shown in Figure 1(f). This shows that a decreased CT exposure can be simulated through our augmentation process for specific CT imaging systems and kernels.

# Acknowledgements

Jacob McIntosh was supported by an appointment to the Research Participation Program at the Center for Devices and Radiological Health administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and the U.S. Food and Drug Administration. The mention of softwares, commercial products, their sources, or their use in connection with material reported herein is not to be construed as either an actual or implied endorsement of such products by the Department of Health and Human Services.

We would also like to express our gratitude to the developers of Albumentations for their excellent work on the original library. The pipeline, design, and API - originally introduced in the Albumentations library - served as the foundation for DICaugment.

# References