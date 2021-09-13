import json
import os
import sys

import cv2
import numpy as np

if __name__ == "__main__":
    INPUT_DIR = "/opt/ml/processing/input/data/"
    OUTPUT_DIR = "/opt/ml/processing/output/test/"

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    crop_width = int(args["crop_width"])
    crop_height = int(args["crop_height"])
    input_width = int(args["input_width"])
    input_height = int(args["input_height"])
    n_classes = int(args["n_classes"])

    H = max([int(output[5:12]) for output in os.listdir(INPUT_DIR)]) + crop_height
    W = max([int(output[13:20]) for output in os.listdir(INPUT_DIR)]) + crop_width

    prob = np.zeros((H, W, n_classes), dtype=np.uint16)
    for output in os.listdir(INPUT_DIR):
        with open(os.path.join(INPUT_DIR, output), "r") as f:
            crop_prob = np.array(json.load(f)["predictions"][0])
        crop_prob = np.round(crop_prob * 100).astype(np.uint8)
        crop_prob = cv2.resize(crop_prob, (crop_width, crop_height))
        top = int(output[5:12])
        left = int(output[13:20])
        prob[top : top + crop_height, left : left + crop_width, :] += crop_prob

    mask = np.argmax(prob, axis=-1).astype(np.uint8)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask.png"), mask)
