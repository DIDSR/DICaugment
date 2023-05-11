# DONE:
* pytorch/transforms -> to_tensor change
* tensorflow/transforms -> to_tensor (transpose)
* Normalize
* Core
* Blur
* Downscale
* Equalize
* FromFloat
* GaussNoise
* GaussianBlur
* InvertImg
* MedianBlur
* MultiplicativeNoise
* RandomBrightnessContrast (ColorJitter?)
* Sharpen
* ToFloat
* UnsharpMask
* PixelDropout

# ADD:
* Datasets Class and Subclasses


# ALTER:

* **ColorJitter** - Remove Hue and Saturation
* RandomGamma
* Affine
* BBoxSafeRandomCrop	
* CenterCrop
* Crop
* CropAndPad
* ElasticTransform
* Flip
* HorizontalFlip
* NoOp
* LongestMaxSize
* PadIfNeeded
* Perspective
* RandomCrop
* RandomCropFromBorders
* RandomCropNearBBox
* RandomGridShuffle
* RandomResizedCrop
* RandomRotate90
* RandomScale
* RandomSizedBBoxSafeCrop	
* RandomSizedCrop
* Resize
* Rotate
* SafeRotate
* ShiftScaleRotate
* SmallestMaxSize
* Transpose
* VerticalFlip


# UNKNOWN:

* FDA
* CLAHE
* HistogramMatching
* TemplateTransform
* PiecewiseAffine
* RandomToneCurve


# REMOVE:

* MotionBlur
* Emboss
* PixelDistributionAdaptation
* RingingOvershoot
* Superpixels
* ImageCompression
* GlassBlur
* ISONoise
* AdvancedBlur
* ToGray
* ToRGB
* ToSepia
* RGBShift
* HueSaturationValue
* ChannelShuffle
* FancyPCA
* ChannelDropout
* Defocus
* Posterize
* RandomFog
* RandomGravel
* RandomRain
* RandomShadow
* RandomSnow
* RandomSunFlare
* Spatter
* ZoomBlur
* CoarseDropout
* CropNonEmptyMaskIfExists
* GridDistortion
* GridDropout
* MaskDropout
* OpticalDistortion
* Solarize
* Lambda
