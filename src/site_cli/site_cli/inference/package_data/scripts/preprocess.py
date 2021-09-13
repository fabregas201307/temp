"""
This script is used by SageMaker Processing to preprocess input to a U-Net model
"""
import os
import sys
import time

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tifffile as tif  # noqa: E402

if __name__ == "__main__":
    print("script starts")
    INPUT_DIR = "/opt/ml/processing/input/data/0/"
    OUTPUT_DIR = "/opt/ml/processing/output/data/0/"

    print("load image")
    tic = time.time()
    img_file = os.listdir(INPUT_DIR)[0]
    if img_file[-4:] == ".tif":
        img = tif.imread(os.path.join(INPUT_DIR, img_file))
        img = img[:, :, :-1]
    else:
        img = cv2.imread(os.path.join(INPUT_DIR, img_file), cv2.IMREAD_COLOR)
    toc = time.time()
    print(f"image loaded ({toc - tic} s)")
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
            0, W - crop_width, (int(W / crop_width) + 1) * overlap_factor
        )
    ]
    tic = time.time()
    print("nested for-loop starts")
    for i in crop_tops:
        for j in crop_lefts:
            crop_fname = f"crop_{i:07d}x{j:07d}.png"
            crop = cv2.resize(
                img[i : i + crop_height, j : j + crop_width, :],
                (input_width, input_height),
            )
            cv2.imwrite(os.path.join(OUTPUT_DIR, crop_fname), crop)
    toc = time.time()
    print(f"nested for-loop ends  ({toc - tic} s)")

    print("script ends")
