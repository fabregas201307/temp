"""
This script is used by SageMaker Processing to postprocess output from a U-Net model
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if __name__ == "__main__":
    print("script starts")
    INPUT_DIR = "/opt/ml/processing/input/data/0/"
    OUTPUT_DIR = "/opt/ml/processing/output/data/0/"

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    crop_width = int(args["crop_width"])
    crop_height = int(args["crop_height"])
    input_width = int(args["input_width"])
    input_height = int(args["input_height"])
    classes = args["classes"].split(",") + ["others"]
    n_classes = len(classes)

    H = max([int(output[5:12]) for output in os.listdir(INPUT_DIR)]) + crop_height
    W = max([int(output[13:20]) for output in os.listdir(INPUT_DIR)]) + crop_width

    tic = time.time()
    print("parallel for-loop starts")
    prob = np.zeros((H, W, n_classes), dtype=np.uint16)

    def load_output(output):
        with open(os.path.join(INPUT_DIR, output), "r") as f:
            crop_prob = np.array(json.load(f)["predictions"][0])
        crop_prob = np.round(crop_prob * 100).astype(np.uint8)
        return crop_prob

    executor = ProcessPoolExecutor()
    crops_prob = executor.map(load_output, sorted(os.listdir(INPUT_DIR)))
    toc = time.time()
    print(f"parallel for-loop ends ({toc - tic} s)")

    print("for-loop starts")
    tic = time.time()
    for crop_prob, output in zip(crops_prob, sorted(os.listdir(INPUT_DIR))):
        crop_prob = cv2.resize(crop_prob, (crop_width, crop_height))
        top = int(output[5:12])
        left = int(output[13:20])
        prob[top : top + crop_height, left : left + crop_width, :] += crop_prob
    toc = time.time()
    print(f"for-loop ends ({toc - tic} s)")

    mask = np.argmax(prob, axis=-1).astype(np.uint8)
    print("write image")
    tic = time.time()
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask.png"), mask)
    toc = time.time()
    print(f"image written ({toc - tic})")

    print("script ends")
