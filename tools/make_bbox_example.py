import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dicaugment as A
import random
from PIL import Image
import glob
import cv2

import shutil



def gif_ify(directory, ext = ".png"):
        frames = [Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))]
        frames_rev = [Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))][1:-1][::-1]
        first_frame = frames[0]
        first_frame.save("bbox_example.gif", append_images = frames[1:] + frames_rev, optimize= False, save_all= True, duration = 100, loop = 0)


def main():

    os.chdir("./tools")

    if os.path.exists("./out"):
        shutil.rmtree("./out")

    os.mkdir("./out")

    img = []
    vmin = 32500
    vmax = 33150

    for file in sorted(glob.glob("bbox_example_data_000088_03_01/*.png")):
        tmp = cv2.imread(file, -1).astype(np.float32)

        img.append(tmp)
        

    img = np.stack(img, axis=2)

    img = np.clip(img, vmin, vmax)
    img -= vmin
    img /= (vmax - vmin)

    bboxes = [[265,211,6,310,257,14]]


    tranforms = [
        A.NoOp(p=1),
        A.RandomSizedCrop(min_max_height= (420,420), height=512, depth=21, width=512, d2h_ratio=0.05, p=1),
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.ShiftScaleRotate(p=1,value = 0, mask_value= 0, shift_limit_z=0, scale_limit=(0, 0.0625))
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
        aug = A.Compose([t], bbox_params= {"format": "pascal_voc_3d"})
        result = aug(image = img, bboxes = bboxes)
        out.append((result["image"], result["bboxes"]))


    for i in range(img.shape[2]):
        fig, axes = plt.subplots(1, 6, figsize= (20,6))

        plt.subplots_adjust(wspace=0.05, hspace=0.02)

        for _,(name, im_mask) in enumerate(zip(labels, out)):

            arr, bboxes_out = im_mask
            x_min, y_min, z_min, x_max, y_max, z_max = [int(p) for p in bboxes_out[0]]
            im_out = cv2.cvtColor(arr[...,i], cv2.COLOR_GRAY2RGB)

            if i >= z_min and i <= z_max:
                axes[_].imshow(cv2.rectangle(im_out, (x_min, y_min), (x_max, y_max), color=(0, 1, 0), thickness=2), vmin =0, vmax = 1)
            else:
                axes[_].imshow(im_out, vmin =0, vmax = 1)
            # axes[1][_].imshow(rgb_img)
            axes[_].set_title(name)
            axes[_].set_axis_off()
            # axes[1][_].set_axis_off()


        fig.savefig("./out/im_{:02d}.png".format(i), bbox_inches = "tight", dpi = 200)

        plt.close(fig)

    
    gif_ify("./out/")


    shutil.rmtree("./out")


if __name__ == "__main__":
     main()