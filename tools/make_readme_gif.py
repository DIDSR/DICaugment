from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dicaugment as A
import random
from PIL import Image
import glob

import shutil



def main():

    os.chdir("./tools")

    if os.path.exists("./out"):
        shutil.rmtree("./out")

    os.mkdir("./out")


    img = np.load("100040_T1F2_normalized.npy")
    img = np.clip(img, -1000, 500)


    nps_normed = np.load("npsNoise_normalized.npy")
    guassian_normed = np.load("GaussianNoise_normalized.npy")


    tranforms = [
        A.NoOp(p=1),
        A.Blur((5,5), p = 1, mode='nearest'),
        A.MedianBlur((5,5), p = 1),
        A.CoarseDropout(p = 1, max_holes= 30, fill_value=-1000),
        A.UnsharpMask(blur_limit=3, threshold=0, alpha =(1.0,1.0), mode='nearest', p = 1),
        A.RandomSizedCrop(min_max_height= (50,80), height=128, width=128, depth=82, p=1),
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.Downscale(scale_max= 0.5, scale_min=0.5, p=1, interpolation= 2),
        A.RandomBrightnessContrast(max_brightness=1000, p=1),
        A.RandomRotate90(p=1),
        A.Sharpen(alpha= (0.05, 0.1), lightness= (0.1, 0.2), mode = 'nearest', p=1)
    ]

    labels = [
        "Original Image",
        "Blur",
        "MedianBlur",
        "CoarseDropout",
        "GuassianNoise",
        "RandomSizedCrop",
        "VerticalFlip",
        "HorizontalFlip",
        "Downscale",
        "NPSNoise",
        "RandomRotate90",
        "Sharpen",
    ]


    out = []

    random.seed(420)
    np.random.seed(420)
    for _, t in enumerate(tranforms):
        aug = A.Compose([t])
        out.append(np.clip(aug(image = img)["image"], -1000, 1000))


    for i in range(img.shape[2]- 14):
        fig, axes = plt.subplots(3,4, figsize= (10,8))

        for _,(name, arr) in enumerate(zip(labels, out)):

            if _ == 4:
                 axes.flat[_].imshow(guassian_normed[...,i], cmap='gray', vmin=-800, vmax=1000)
            elif _ == 9:
                 axes.flat[_].imshow(nps_normed[...,i], cmap='gray', vmin=-1100, vmax=1000)
            else:
                axes.flat[_].imshow(arr[...,i], cmap='gray', vmin=-1000, vmax=1000)
            axes.flat[_].set_title(name)
            axes.flat[_].set_axis_off()

        fig.savefig("out/im_{:02d}.png".format(i), bbox_inches = "tight", dpi = 90)

        plt.close(fig)

    
    gif_ify("./out/")


    shutil.rmtree("./out")


    




def gif_ify(directory, ext = ".png"):
        frames = [Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))]
        first_frame = frames[0]
        first_frame.save("README_example.gif", format="GIF", append_images =frames[1:], save_all= True, duration = 100, loop = 0)


if __name__ == "__main__":
     main()