"""
This script is used by SageMaker Processing to postprocess output from a U-Net model
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

    INPUT_IMAGE_DIR = "/opt/ml/processing/input/data/0/"
    P1_MASK_DIR = "/opt/ml/processing/input/data/1/"
    P2_MASK_DIR = "/opt/ml/processing/input/data/2/"
    OUTPUT_DIR = "/opt/ml/processing/output/data/0/"

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    p1_classes = args["p1_classes"].split(",") + ["others"]
    p2_classes = args["p2_classes"].split(",") + ["others"]

    img_file = os.listdir(INPUT_IMAGE_DIR)[0]
    p1_mask_file = os.listdir(P1_MASK_DIR)[0]
    p2_mask_file = os.listdir(P2_MASK_DIR)[0]

    print("load images")
    tic = time.time()
    if img_file[-4:] == ".tif":
        img = tif.imread(os.path.join(INPUT_IMAGE_DIR, img_file))
        mask_input = img[:, :, -1]
        img = img[:, :, :-1]
    else:
        img = cv2.imread(os.path.join(INPUT_IMAGE_DIR, img_file), cv2.IMREAD_COLOR)
        mask_input = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask_p1 = cv2.imread(os.path.join(P1_MASK_DIR, p1_mask_file), cv2.IMREAD_GRAYSCALE)
    mask_p2 = cv2.imread(os.path.join(P2_MASK_DIR, p2_mask_file), cv2.IMREAD_GRAYSCALE)
    toc = time.time()
    print(f"images loaded ({toc - tic} s)")

    print("merge masks")
    tic = time.time()
    mask = mask_p2
    mask[
        (mask_input == 0)
        | (
            (mask_p1 != p1_classes.index("asphalt"))
            & (mask_p1 != p1_classes.index("concrete"))
        )
    ] = (len(p2_classes) - 1)
    toc = time.time()
    print(f"masks merged ({toc - tic} s)")

    print("write image")
    tic = time.time()
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask.png"), mask)
    toc = time.time()
    print(f"image written ({toc - tic} s)")

    # Saeed's code here
    # get_measurement(mask, os.path.join(INPUT_IMAGE_DIR, img))

    print("script ends")
