import sys
from pathlib import Path

import boto3
import cv2

from site_tools import call_endpoint_with_local_image, get_image_files

if __name__ == "__main__":
    # Get only jpg image files from passed in folder input
    files = get_image_files("/opt/ml/processing/input/data/", [".jpg"])
    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    endpoint_name = args.get("endpoint", "ss-labelbox-1024-size-1024-v3")
    chipsize = int(args.get("chipsize", 1024))
    crf_post_process = args.get("crf_post_process", False)
    verbose = args.get("verbose", False)
    if verbose:
        if verbose == "False":
            verbose = False
        else:
            verbose = True

    client = boto3.client("runtime.sagemaker", region_name="us-east-2")

    for filename in files:

        img_rebuilt = call_endpoint_with_local_image(
            filename, client, endpoint_name, chipsize, crf_post_process
        )

        filewrite = Path(filename).stem
        if verbose:
            print(str(filewrite), img_rebuilt.shape)
        cv2.imwrite(
            "/opt/ml/processing/output/train/" + filewrite + ".png", img_rebuilt
        )
