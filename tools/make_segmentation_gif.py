from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
import cv2

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

    img  = np.rot90(nib.load("./amos_data/amos_0024_image.nii.gz").get_fdata()).astype(np.float32)
    mask = np.rot90(nib.load("./amos_data/amos_0024_label.nii.gz").get_fdata()).astype(np.uint8)

    vmin = -1024
    vmax = 750

    img = np.clip(img, vmin, vmax)
    img -= img.min()
    img /= img.max()

    cm = plt.get_cmap('gist_rainbow')
    color_map = np.array([(0,0,0,0)]+ list(map(cm, np.linspace(0,0.9, 15)))).astype("float32")[:,:3]

    tranforms = [
        A.NoOp(p=1),
        A.RandomSizedCrop(min_max_height= (400,450), height=512, depth=85, width=512, d2h_ratio=0.15, p=1),
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.ShiftScaleRotate(p=1, value = 0, mask_value= 0, shift_limit_z=0)
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
    random.seed(43)
    np.random.seed(43)
    for _, t in enumerate(tranforms):
        aug = A.Compose([t])
        result = aug(image = img, mask = mask)
        out.append((result["image"], result["mask"]))


    for i in range(4, img.shape[2]- 4):
        fig, axes = plt.subplots(2, 6, figsize= (20,6))

        plt.subplots_adjust(wspace=0.05, hspace=0.02)

        for _,(name, im_mask) in enumerate(zip(labels, out)):

            arr, mask_arr = im_mask

            axes[0][_].imshow(arr[...,i], cmap='gray', vmin =0, vmax = 1)
            rgb_img = cv2.cvtColor(arr[...,i], cv2.COLOR_GRAY2RGB)
            rgb_mask = color_map[mask_arr[...,i]]
            rgb_img[rgb_mask != 0] = rgb_mask[rgb_mask != 0]
            axes[1][_].imshow(rgb_img)
            axes[0][_].set_title(name)
            axes[0][_].set_axis_off()
            axes[1][_].set_axis_off()

        

        fig.savefig("out/im_{:02d}.png".format(i), bbox_inches = "tight", dpi = 200)

        plt.close(fig)

    
    gif_ify("./out/")


    shutil.rmtree("./out")


    
def gif_ify(directory, ext = ".png"):
        frames = [Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))]
        first_frame = frames[0]
        first_frame.save("segmentation_example.gif", format="GIF", append_images = frames[1:] + frames[1:-1][::-1], save_all= True, duration = 100, loop = 0)


if __name__ == "__main__":
     main()