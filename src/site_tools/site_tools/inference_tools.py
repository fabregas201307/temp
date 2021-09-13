import io
from pathlib import Path
from typing import Iterator, List, Union

import boto3
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from pydensecrf.utils import unary_from_labels


def chip_iterator(image_path: Union[str, Path], size: int = 1024) -> Iterator[tuple]:
    """Generator that yields chips of size `size`x`size from `image`.

    Parameters
    ----------
    image_path: file path to image.
    size: size of broken up images. The default is 1024.
    folders: Folders to include when recursing. None (default) searches all folders.

    Returns
    -------
    An iterator of tuples that returns a (chip, top left x-coordinate of the current
    chip, top left y-coordinate of the current chip, and the original shape of the
    image).
    """
    img = cv2.imread(str(image_path))
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
    """ Regenerates a full image from chip predictions.

    Parameters
    ----------
    chip_preds: List of numpy arrays of chip annotation predictions
    xis: list of top left x-coordinate for each chip
    yis: list of top left y-coordinate for each chip
    shape: original shape of the image

    Returns
    -------
    A numpy array of the full image annotation rebuilt from the individual chips.
    """

    sizex = chip_preds[0].shape[0]
    sizey = chip_preds[0].shape[1]
    full_annotation = np.zeros(orig_image_shape[:2], dtype="uint8")

    for chip_pred, x, y in zip(chip_preds, xis, yis):
        section = full_annotation[y : y + sizey, x : x + sizex].shape
        full_annotation[y : y + sizey, x : x + sizex] = chip_pred[
            : section[0], : section[1]
        ]
    return full_annotation


def call_endpoint_with_local_image(
    image_path: Union[Path, str],
    sagemaker_client: boto3.session.Session.client,
    endpoint_name: str,
    chipsize: int = 1024,
    crf_post_process: bool = False,
) -> np.array:
    """Loads in a local image and gets the annotation from the endpoint.

    Parameters
    ----------
    image_path: path to local image file
    sagemaker_client: boto3 sagemaker client
    endpoint_name: the name of the endpoint to call
    chipsize: size of the image chips. Default is 1024.
    crf_post_process: Do crf post processing of image. Default is False.

    Returns
    -------
    A numpy array of the full image annotation rebuilt from the individual chips.
    """
    # Get iterator with each image in a reduced size
    chips = chip_iterator(image_path, size=chipsize)
    chip_preds = []
    xs = []
    ys = []
    # Loop through each image calling the endpoint and saving the prediction
    for (chip, x, y, _shape) in chips:
        # load image as jpg and convert to bytearray format for sagemaker
        success, encoded_image = cv2.imencode(".jpg", chip)
        response = sagemaker_client.invoke_endpoint(
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
        if crf_post_process:
            chip_pred = do_crf(chip, np.array(chip_pred), zero_unsure=False)
        chip_preds.append(np.array(chip_pred))
        xs.append(x)
        ys.append(y)

    # rebuild the full image from chips
    img_rebuilt = full_image_from_chip_preds(chip_preds, xs, ys, _shape)
    return img_rebuilt


def compute_iou(actual: np.array, pred: np.array, num_cls: int = 6) -> np.array:
    """To calculate class iou score for an image.

    Args:
        actual (image array): annotated image mask by human
        pred (image array): predicted image mask by model
        num_cls (int, optional): Number of classes in mask. Defaults to 6.

    Returns:
        [iou]: Intersection over union score as numpy array.
    """
    a = actual
    # if background class (255) not remaped, makes it equal to the last class
    a = np.where(a > num_cls, num_cls - 1, a)
    a = a.flatten()
    a_count = np.bincount(a, weights=None, minlength=num_cls)  # A

    b = pred
    # if background class (255) not remaped, makes it equal to the last class
    b = np.where(b > num_cls, num_cls - 1, b)
    b = b.flatten()
    b_count = np.bincount(b, weights=None, minlength=num_cls)  # B

    c = a * num_cls + b
    cm = np.bincount(c, weights=None, minlength=num_cls ** 2)
    cm = cm.reshape((num_cls, num_cls))

    Nr = np.diag(cm)  # A ⋂ B
    Dr = a_count + b_count - Nr  # A ⋃ B
    individual_iou = Nr / Dr
    miou = np.nanmean(individual_iou)

    return miou, list(np.around(individual_iou, decimals=2))


def do_crf(im: np.array, mask: np.array, zero_unsure: bool = True):
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
