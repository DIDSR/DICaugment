# ADD:

* pytorch/transforms -> to_tensor change
* tensorflow/transforms -> to_tensor (transpose)
* Datasets Class and Subclasses


# ALTER:

* AdvancedBlur
* Blur
* CLAHE
* **ColorJitter** - Remove Hue and Saturation
* Downscale
* Emboss
* Equalize
* FromFloat
* GaussNoise
* GaussianBlur
* InvertImg
* MedianBlur
* MotionBlur
* MultiplicativeNoise
* Normalize
* RandomBrightnessContrast (ColorJitter?)
* RandomGamma
* RandomToneCurve
* Sharpen
* Solarize
* ToFloat
* UnsharpMask
* Affine
* BBoxSafeRandomCrop	
* CenterCrop
* Crop
* CropAndPad
* ElasticTransform
* Flip
* HorizontalFlip
* Lambda
* LongestMaxSize
* NoOp
* PadIfNeeded
* Perspective
* PixelDropout
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
* GlassBlur
* HistogramMatching
* ISONoise
* ImageCompression
* PixelDistributionAdaptation
* RingingOvershoot
* Superpixels
* TemplateTransform
* PiecewiseAffine

# REMOVE:

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