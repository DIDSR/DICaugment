from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import albumentations3d as A
import random
from PIL import Image
import glob

import shutil



def main():

    os.chdir("./tools")

    if os.path.exists("./out"):
        shutil.rmtree("./out")

    os.mkdir("./out")


    img = np.load("100040_T1F2.npy")
    img = np.clip(img, -1000, 2000)


    tranforms = [
        A.NoOp(p=1),
        A.Blur((5,5), p = 1),
        A.MedianBlur((5,5), p = 1),
        A.CoarseDropout(p = 1, max_holes= 20, fill_value=-1000),
        A.GridDropout(shift_x= 0, shift_y= 0, shift_z=0, p=1, fill_value=-1000),
        A.RandomSizedCrop(min_max_height= (50,80), height=101, depth=41, width=101, d2h_ratio=0.5, p=1),
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.Downscale(p=1, interpolation= 2),
        A.RandomBrightnessContrast(max_brightness=2000, p=1),
        A.RandomRotate90(p=1),
        A.Sharpen(p=1)
    ]

    labels = [
        "Original Image",
        "Blur",
        "MedianBlur",
        "CoarseDropout",
        "GridDropout",
        "RandomSizedCrop",
        "VerticalFlip",
        "HorizontalFlip",
        "Downscale",
        "BrightnessContrast",
        "RandomRotate90",
        "Sharpen",
    ]


    out = []

    random.seed(420)
    np.random.seed(420)
    for _, t in enumerate(tranforms):
        aug = A.Compose([t])
        out.append(np.clip(aug(image = img)["image"], -1000, 2000))


    for i in range(41):
        fig, axes = plt.subplots(3,4, figsize= (10,8))

        for _,(name, arr) in enumerate(zip(labels, out)):

            axes.flat[_].imshow(arr[...,i], cmap='gray', vmin=-1000, vmax=2000)
            axes.flat[_].set_title(name)
            axes.flat[_].set_axis_off()

        fig.savefig("out/im_{}.png".format(i), bbox_inches = "tight", dpi = 200)

        plt.close(fig)

    
    gif_ify("./out/")


    shutil.rmtree("./out")


    




def gif_ify(directory, ext = ".png"):
        frames = [Image.open(im) for im in glob.glob("{}*{}".format(directory, ext))]
        first_frame = frames[0]
        first_frame.save("README_example.gif", format="GIF", append_images = frames[1:], save_all= True, duration = 400, loop = 0)


if __name__ == "__main__":
     main()