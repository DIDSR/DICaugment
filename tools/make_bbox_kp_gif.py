import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dicaugment as A
import random
from PIL import Image
import glob

import shutil


def get_cube_edge_pairs(bbox):
    min_x, min_y, min_z, max_x, max_y, max_z = bbox

    # Generate all edge pairs
    edge_pairs = [
        [(min_x, min_y, min_z), (min_x, min_y, max_z)],
        [(min_x, min_y, min_z), (min_x, max_y, min_z)],
        [(min_x, min_y, min_z), (max_x, min_y, min_z)],
        [(min_x, min_y, max_z), (min_x, max_y, max_z)],
        [(min_x, min_y, max_z), (max_x, min_y, max_z)],
        [(min_x, max_y, min_z), (min_x, max_y, max_z)],
        [(min_x, max_y, min_z), (max_x, max_y, min_z)],
        [(min_x, max_y, max_z), (max_x, max_y, max_z)],
        [(max_x, min_y, min_z), (max_x, min_y, max_z)],
        [(max_x, min_y, min_z), (max_x, max_y, min_z)],
        [(max_x, min_y, max_z), (max_x, max_y, max_z)],
        [(max_x, max_y, min_z), (max_x, max_y, max_z)],
    ]

    return edge_pairs


def gif_ify(directory, ext=".png"):
    frames = [
        Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))
    ]
    frames_rev = [
        Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))
    ][1:-1][::-1]
    first_frame = frames[0]
    first_frame.save(
        "bbox_kp_example.gif",
        append_images=frames[1:] + frames_rev,
        optimize=False,
        save_all=True,
        duration=100,
        loop=0,
    )


def main():
    os.chdir("./tools")

    if os.path.exists("./out"):
        shutil.rmtree("./out")

    os.mkdir("./out")

    image = np.zeros((100, 100, 100, 3), dtype=np.uint8)

    for i in range(10, 40):
        image[i : 80 - i, 80 - i - 2 : 80 - i, 20 + i] = [255, 0, 255]
        image[i : 80 - i, i : i + 2, 20 + i] = [255, 255, 0]

        image[80 - i - 2 : 80 - i, i : 80 - i, 20 + i] = [0, 0, 255]
        image[i : i + 2, i : 80 - i, 20 + i] = [0, 255, 0]

    bboxes = [[9, 9, 29, 71, 71, 61]]
    keypoints = [[10, 10, 30], [70, 70, 30], [10, 70, 30], [70, 10, 30], [40, 40, 59]]

    random.seed(42)
    np.random.seed(42)

    aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=(-0.1, -0.1),
                rotate_limit=(45, 45),
                axes="yz",
                interpolation=0,
                value=0,
                rotate_method="ellipse",
            )
        ],
        bbox_params={"format": "pascal_voc_3d"},
    )

    response = aug(image=image, bboxes=bboxes)
    bb_im = response["image"]
    new_bboxes = response["bboxes"]

    aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(0.1, 0.1),
                scale_limit=(0.1, 0.1),
                rotate_limit=(30, 30),
                axes="xz",
                interpolation=0,
                value=0,
            )
        ],
        keypoint_params={"format": "xyz"},
    )

    response = aug(image=image, keypoints=keypoints)
    kp_im = response["image"]
    new_keypoints = response["keypoints"]

    labels = ["Original Image", "Augmented Image"]

    for i in range(30, 60):
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 10), subplot_kw=dict(projection="3d")
        )

        plt.subplots_adjust(wspace=0.05, hspace=0.1)

        for _, (im, bb) in enumerate([(image, bboxes), (bb_im, new_bboxes)]):
            # Get the coordinates and color values of non-zero pixels
            x, y, z = np.nonzero(np.any(im != 0, axis=3))
            colors = im[x, y, z]

            # Plot the pixels as a scatter plot
            axes[0][_].scatter(x, y, z, c=colors / 255.0, s=1.5)

            for edgepair in get_cube_edge_pairs(bb[0]):
                v1 = edgepair[0]
                v2 = edgepair[1]

                EdgeXvals = [v1[0], v2[0]]
                EdgeYvals = [v1[1], v2[1]]
                EdgeZvals = [v1[2], v2[2]]

                x = np.linspace(v1[0], v2[0], 100)
                y = np.linspace(v1[1], v2[1], 100)
                z = np.linspace(v1[2], v2[2], 100)

                axes[0][_].scatter(
                    x,
                    y,
                    z,
                    c="red",
                    s=0.1,
                )

            # Set plot limits and labels
            axes[0][_].set_xlim(0, im.shape[0])
            axes[0][_].set_ylim(0, im.shape[1])
            axes[0][_].set_zlim(0, im.shape[2])
            axes[0][_].set_title(labels[_])
            axes[0][_].set_axis_off()
            axes[0][_].set_facecolor("black")
            axes[0][_].view_init(elev=45, azim=i, roll=0)

        for _, (im, kp) in enumerate([(image, keypoints), (kp_im, new_keypoints)]):
            # Get the coordinates and color values of non-zero pixels
            x, y, z = np.nonzero(np.any(im != 0, axis=3))
            colors = im[x, y, z]

            # Plot the pixels as a scatter plot
            axes[1][_].scatter(x, y, z, c=colors / 255.0, s=1.5, zorder=2)

            kp = np.array(kp)

            x, y, z = kp[:, 0], kp[:, 1], kp[:, 2]

            axes[1][_].scatter(x, y, z, c="red", s=70, alpha=1, zorder=1)

            # Set plot limits and labels
            axes[1][_].set_xlim(0, im.shape[0])
            axes[1][_].set_ylim(0, im.shape[1])
            axes[1][_].set_zlim(0, im.shape[2])
            axes[1][_].set_title(labels[_])
            axes[1][_].set_axis_off()
            axes[1][_].set_facecolor("black")
            axes[1][_].view_init(elev=45, azim=i, roll=0)

        fig.savefig("./out/im_{}.png".format(i), bbox_inches="tight", dpi=200)

        plt.close(fig)

    gif_ify("./out/")

    shutil.rmtree("./out")


if __name__ == "__main__":
    main()
