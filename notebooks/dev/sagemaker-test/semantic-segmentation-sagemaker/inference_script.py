import io
import mimetypes
import os
import sys
from pathlib import Path
from typing import Iterator, List, Union

import boto3
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from pydensecrf.utils import unary_from_labels
from tqdm import tqdm


def _get_files(
    pathname: Union[str, Path],
    filenames: list,
    extensions: Union[set, list, None] = None,
):
    """A helper for `get_files` to generate the full paths and include only files
    within `extensions`.

    Parameters
    ----------
    pathname: Path of directory to retrieve files.
    filenames: Filenames in the `pathname` directory.
    extensions: Extensions to include in the search.
        None (default) will select all extensions.

    Returns
    -------
    A list of full pathnames with only `extensions`, if included.
    """
    pathname = Path(pathname)
    res = [
        pathname / f
        for f in filenames
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path: Union[Path, str],
    extensions: Union[set, list] = None,
    recurse: bool = True,
    folders: list = None,
) -> list:
    """Get all the files in `path` with optional `extensions`, optionally with
    `recurse`, only in `folders`, if specified.

    Parameters
    ----------
    path: Base path to directory to retrieve files.
    extensions: Extensions to include in the search.
        None (default) will select all extensions.
    recurse: Indicating whether to recursively search the directory, defaults to True.
    folders: Folders to include when recursing. None (default) searches all folders.

    Returns
    -------
    A list of files with `extensions` and in these `folders`, if specified.
    """
    if not folders:
        folders = []
    path = Path(path)
    if extensions:
        extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return res


def get_image_files(
    path: Union[Path, str],
    extensions: Union[set, list] = None,
    recurse: bool = True,
    folders: list = None,
) -> list:
    """Get image files in `path` recursively, only in `folders`, if specified.

    Parameters
    ----------
    path: Base path to image directory.
    extensions: Extensions to include in the search.
        None (default) will select all mimetypes with `image/`.
    recurse: Indicating whether to recursively search the directory, defaults to True.
    folders: Folders to include when recursing. None (default) searches all folders.

    Returns
    -------
    A list of image files.
    """
    if not extensions:
        extensions = get_image_extensions()
    return get_files(path, extensions=extensions, recurse=recurse, folders=folders)


def get_image_extensions() -> set:
    """Get standard mimetype image extensions available on the system.

    Returns
    -------
    Set of image extensions.
    """
    extensions = set(
        k for k, v in mimetypes.types_map.items() if v.startswith("image/")
    )
    return extensions


def chip_iterator(image: Union[str, Path], size: int = 1024) -> Iterator[tuple]:
    """Generator that yields chips of size `size`x`size from `image`."""

    img = cv2.imread(str(image))
    shape = img.shape

    for x in range(0, shape[1], size):
        for y in range(0, shape[0], size):
            chip = img[y : y + size, x : x + size, :]
            # Padding right and bottom out to `size` with black pixels
            chip = cv2.copyMakeBorder(
                chip,
                top=0,
                bottom=size - chip.shape[0],
                left=0,
                right=size - chip.shape[1],
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            # So chip yields rgb image instead of bgr
            yield (chip[:, :, ::-1], x, y, shape)


def full_image_from_chip_preds(
    chip_preds: List[np.array], xis: List[int], yis: List[int], orig_image_shape: tuple,
) -> np.array:
    """ Regenerates a full image from chip predictions."""

    sizex = chip_preds[0].shape[0]
    sizey = chip_preds[0].shape[1]
    full_annotation = np.zeros(orig_image_shape[:2], dtype="uint8")

    for chip_pred, x, y in zip(chip_preds, xis, yis):
        section = full_annotation[y : y + sizey, x : x + sizex].shape
        full_annotation[y : y + sizey, x : x + sizex] = chip_pred[
            : section[0], : section[1]
        ]
    return full_annotation


def do_crf(im, mask, zero_unsure=True):
    """Perform crf on predicted mask.

    Condition Random Fields is are a class of statistical modeling method often applied
    in pattern recognition and machine learning and used for structured prediction.
    Whereas a classifier predicts a label for a single sample without considering
    "neighboring" samples, a CRF can take context into account

    Args:
        im (np.array): original image.
        mask (np.array): predicted mask from the model.
        zero_unsure (bool, optional): Always set to True Defaults to True.

    Returns:
        image (np.array): predicted with crf implemented.

    """
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    # In case of single class in image there is no need of crf
    if n_labels == 1:
        return mask
    # print(n_labels)
    d = dcrf.DenseCRF2D(
        image_size[1], image_size[0], n_labels
    )  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype("uint8"), compat=10)
    Q = d.inference(5)  # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map:  # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP
    # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)


if __name__ == "__main__":
    # Get only jpg image files from passed in folder input
    files = get_image_files("/opt/ml/processing/input/data/", [".jpg"])

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    endpoint_name = args.get("endpoint")
    crf_post_process = args.get("crf_post_process", True)
    verbose = args.get("verbose", False)
    chipsize = int(args.get("chipsize", 1024))

    client = boto3.client("runtime.sagemaker", region_name="us-east-2")

    for filename in tqdm(files, total=len(files)):
        # Get iterator with each image in a reduced size
        chips = chip_iterator(filename, size=chipsize)
        chip_preds = []
        xs = []
        ys = []
        # Loop through each image calling the endpoint and saving the prediction
        for (chip, x, y, _shape) in chips:
            # load image as jpg and convert to bytearray format for sagemaker
            success, encoded_image = cv2.imencode(".jpg", chip)
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="image/jpeg",
                Accept="image/png",
                Body=encoded_image.tobytes(),
            )
            # Get response and turn into a numpy array (storing the position of the chip
            # as well)
            response_body = response["Body"].read()
            prediction = io.BytesIO(response_body)
            chip_pred = Image.open(prediction)
            chip_pred = do_crf(chip, np.array(chip_pred), zero_unsure=False)
            chip_preds.append(np.array(chip_pred))
            xs.append(x)
            ys.append(y)

        # rebuild the full image from chips
        img_rebuilt = full_image_from_chip_preds(chip_preds, xs, ys, _shape)

        filewrite = Path(filename).stem
        if verbose:
            print(str(filewrite), img_rebuilt.shape)
        cv2.imwrite(
            "/opt/ml/processing/output/train/" + filewrite + ".png", img_rebuilt
        )
