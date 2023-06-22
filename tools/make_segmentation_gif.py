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
    
    tmp_mask = (img[50:65,40:55,18:22] > -500).astype(np.uint8)
    tmp_mask[:2, :, 3] = 0

    mask_full = np.zeros_like(img, dtype= np.uint8)
    mask_full[50:65,40:55,18:22] = tmp_mask

    img_crop = img[20:80,20:80,15:25]
    mask_crop = mask_full[20:80,20:80,15:25]

    tranforms = [
        A.NoOp(p=1),
        A.RandomSizedCrop(min_max_height= (20,50), height=60, depth=10, width=60, d2h_ratio=0.18, p=1),
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.ShiftScaleRotate(p=1, value = -1000, mask_value= 0)
    ]

    labels = [
        "Original Image",
        "RandomSizedCrop",
        "VerticalFlip",
        "HorizontalFlip",
        "RandomRotate90",
        "ShiftScaleRotate",
    ]

    out = []
    random.seed(42)
    np.random.seed(42)
    for _, t in enumerate(tranforms):
        aug = A.Compose([t])
        result = aug(image = img_crop, masks = [mask_crop])
        out.append((np.clip(result["image"], -1000, 2000), result["masks"][0]))

    
    for i in range(img_crop.shape[2]):
        fig, axes = plt.subplots(2, 6, figsize= (20,6))

        plt.subplots_adjust(wspace=0.05, hspace=0.02)

        for _,(name, im_mask) in enumerate(zip(labels, out)):

            arr, mask = im_mask

            axes[0][_].imshow(arr[...,i], cmap='gray', vmin=-1000, vmax=2000)
            axes[1][_].imshow(mask[...,i], cmap='gray', vmin=0, vmax=1)
            axes[0][_].set_title(name)
            axes[0][_].set_axis_off()
            axes[1][_].set_axis_off()

        

        fig.savefig("out/im_{}.png".format(i), bbox_inches = "tight", dpi = 200)

        plt.close(fig)

    
    gif_ify("./out/")


    shutil.rmtree("./out")


    
def gif_ify(directory, ext = ".png"):
        frames = [Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))]
        first_frame = frames[0]
        first_frame.save("segmentation_example.gif", format="GIF", append_images = frames[1:], save_all= True, duration = 350, loop = 0)


if __name__ == "__main__":
     main()