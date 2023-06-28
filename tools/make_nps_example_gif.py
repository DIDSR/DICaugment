from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import random
from PIL import Image
import glob

import shutil



def main():

    os.chdir("./tools")

    if os.path.exists("./out"):
        shutil.rmtree("./out")

    os.mkdir("./out")

    low_dose = np.load("nps_example_data/simulatedlowdose_std=16.01npy")
    high_dose = np.load("nps_example_data/samplePhantom_240_mAs_std=9.49.npy")
    low_dose_nps = mpimg.imread("nps_example_data/simulated_noise_nps.png")
    high_dose_nps = mpimg.imread("nps_example_data/noise_nps.png")


    for i in range(20):

        fig, axes = plt.subplots(2, 2, figsize= (15,10))

        plt.subplots_adjust(wspace=-0.15, hspace= -0.05)

        vmin = -44
        vmax = 158

        axes[0][0].imshow(high_dose[...,i], vmin= vmin, vmax = vmax, cmap = 'gray')
        axes[0][1].imshow(low_dose[...,i], vmin= vmin, vmax = vmax, cmap = 'gray')
        axes[1][0].imshow(high_dose_nps)
        axes[1][1].imshow(low_dose_nps)

        axes[0][0].set_title("High Dose Phantom, 240 mAs, STD=9.49")
        axes[0][1].set_title("Simulated Low Dose, STD=16.01")
        axes[1][0].set_title("Normalized Radial 1-D NPS", y= 1.0, pad=-25)
        axes[1][1].set_title("Normalized Radial 1-D NPS", y= 1.0, pad=-25)

        for ax in axes.flat:
            ax.set_axis_off()
        


        fig.savefig("./out/im_{:02d}.png".format(i), bbox_inches = "tight", dpi = 200)

        plt.close(fig)

    
    gif_ify("./out/")


    shutil.rmtree("./out")


    
def gif_ify(directory, ext = ".png"):
        frames = [Image.open(im) for im in sorted(glob.glob("{}*{}".format(directory, ext)))]
        first_frame = frames[0]
        first_frame.save("nps_example.gif", append_images = frames[1:], optimize= False, save_all= True, duration = 100, loop = 0)


if __name__ == "__main__":
     main()