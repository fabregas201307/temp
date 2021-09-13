import os
import sys

import cv2
import numpy as np

if __name__ == "__main__":
    INPUT_DIR = "/opt/ml/processing/input/data/"
    OUTPUT_DIR = "/opt/ml/processing/output/test/"

    img_file = os.listdir(INPUT_DIR)[0]
    img = cv2.imread(os.path.join(INPUT_DIR, img_file), cv2.IMREAD_COLOR)
    H, W, _ = img.shape

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    crop_width = int(args["crop_width"])
    crop_height = int(args["crop_height"])
    input_width = int(args["input_width"])
    input_height = int(args["input_height"])
    overlap_factor = int(args["overlap_factor"])

    crop_tops = [
        int(i)
        for i in np.linspace(
            0, H - crop_height, (int(H / crop_height) + 1) * overlap_factor
        )
    ]
    crop_lefts = [
        int(j)
        for j in np.linspace(
            0, W - crop_width, (int(H / crop_width) + 1) * overlap_factor
        )
    ]
    for i in crop_tops:
        for j in crop_lefts:
            crop_fname = f"crop_{i:07d}x{j:07d}.png"
            crop = cv2.resize(
                img[i : i + crop_height, j : j + crop_width, :],
                (input_width, input_height),
            )
            cv2.imwrite(os.path.join(OUTPUT_DIR, crop_fname), crop)
