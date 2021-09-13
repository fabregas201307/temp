import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import correlate1d

from .config import LABELMAP_RGB
from .core_tools import get_files, parallel

Image.MAX_IMAGE_PIXELS = 216000000


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


def verify_image(fn: str) -> bool:
    """Confirm that `fn` can be opened.

    Parameters
    ----------
    fn: Filename of image to be checked if can open

    Returns
    -------
    True, if image can be opened or false.
    """
    try:
        im = Image.open(fn)
        im.draft(im.mode, (32, 32))
        im.load()
        return True
    except Exception:
        return False


def verify_images(fns: list) -> list:
    """Find images in `fns` that can't be opened.

    Parameters
    ----------
    fns: Filenames of images to be checked if can open

    Returns
    -------
    List of filenames of the images that couldn't be opened.
    """
    return [fns[i] for i, o in enumerate(parallel(verify_image, fns)) if not o]


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


def image2tile(
    src_folder_path: Union[str, Path],
    filename: str,
    src_images_path: str = "images",
    src_annotation_path: str = "annotations",
    dest_folder_path: Optional[Path] = None,
    dest_images_path: str = "images",
    dest_annotation_path: str = "annotations",
    windowx: int = 512,
    windowy: int = 512,
    stridex: int = 512,
    stridey: int = 512,
    verbose: bool = False,
) -> None:
    """Tiles a single image and its annotations into seperate images.

    The tiled images are each `windowx` by `windowy`. This saves the tiled images and
    their annotations into their respective folders for processing with sagemaker. NOTE:
    Currently, the last remainder window in the x and y-directions are ignored and
    discarded.

    Parameters
    ----------
    src_folder_path: Source folder that contains the images and annotations
    filename: Filename of image to be tiled (can include extension or no extension)
    src_images_path: Folder path to source images, Default is `images`.
    src_annotation_path: Folder path to source annotations, Default is `annotations`.
    dest_folder_path: Top level path to store the images and annotations. Optional, by
        default is `tiles_<windowx>` at the parent folder of the src_folder_path.
    dest_images_path: Folder path to destination images, Default is `images`.
    dest_annotation_path: Folder path to destination annotations,
        Default is `annotations`.
    windowx: size of the image window in the x-direction, Default is 512.
    windowy: size of the image window in the y-direction, Default is 512.
    stridex: size of the stried in the x-direction, Default is 512.
    stridey: size of the stried in the y-direction, Default is 512.
    verbose: Default is False to not show print statements
    """
    src_folder_path = Path(src_folder_path)
    if dest_folder_path is None:
        # Saves under the parent directory of dest_folder_path as `tiles_` and size of
        # the windowx
        dest_folder_path = src_folder_path.parent / ("tiles_" + str(windowx))
    extensions = get_image_extensions()
    if Path(filename).suffix.lower() in extensions:
        filename_orig = filename
        filename = Path(filename).stem
    else:
        filename_orig = filename + ".JPG"

    image = cv2.imread(str(src_folder_path / src_images_path / (filename_orig)))
    label = cv2.imread(
        str(src_folder_path / src_annotation_path / (filename + ".png")),
        cv2.IMREAD_GRAYSCALE,
    )

    assert (
        image.shape[0] == label.shape[0]
    ), "Image shape[0] is not equal to mask shape[0]."
    assert (
        image.shape[1] == label.shape[1]
    ), "Image shape[1] is not equal to mask shape[1]."

    shape = image.shape

    xsize = shape[1]
    ysize = shape[0]
    if verbose:
        print(f"converting image {filename} {xsize}x{ysize} to tiles ...")

    counter = 0

    for xi in range(0, shape[1] - windowx, stridex):
        for yi in range(0, shape[0] - windowy, stridey):

            imagetile = image[yi : yi + windowy, xi : xi + windowx, :]
            labeltile = label[yi : yi + windowy, xi : xi + windowx]

            imagetile_filepath = dest_folder_path / dest_images_path
            labeltile_filepath = dest_folder_path / dest_annotation_path
            imagetile_filepath.mkdir(parents=True, exist_ok=True)
            labeltile_filepath.mkdir(parents=True, exist_ok=True)
            imagetile_filename = imagetile_filepath / (
                filename + "-" + str(counter).zfill(6) + ".jpg"
            )
            labeltile_filename = labeltile_filepath / (
                filename + "-" + str(counter).zfill(6) + ".png"
            )

            cv2.imwrite(str(imagetile_filename), imagetile)
            cv2.imwrite(str(labeltile_filename), labeltile)
            counter += 1


def category2mask(img: np.array, mask_dict: dict = LABELMAP_RGB) -> np.array:
    """Convert a category image to color mask.

    Parameters
    ----------
    img: Numpy array of an image.
    mask_dict: Dictionary to map values from categories to rgb (or bgr) values.

    Returns
    -------
    Numpy array with category values mapped to easily viewable RGB (or BGR) values from
    `mask_dict`. By default, these are RGB values from the config file.

    """
    # If a 3-channel grayscale image is passed in, it is converted down to one channel
    # before being processed.
    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3,), dtype="uint8")

    for category, mask_color in mask_dict.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask


def get_class_dominated_crops(
    mask_file: str,
    crop_size: Tuple[int, int],
    dominating_class: int,
    num_crop: int = 1,
    threshold: float = 0.8,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Get a set of crops from an image that are dominated by a specified class

    Parameters
    ----------
    mask_file : str
        Path to the file of mask

    crop_size : tuple
        Size of a crop (width, height)

    dominating_class : int
        Index of the class that dominates the crops

    num_crop : int, optional
        Number of crops. Default: 1

    threshold : float, optional
        Fraction of pixels above which a class is considered as dominating a crop.
        Default: 0.8

    seed : int, optional
        Random seed. Default: None.

    Returns
    -------
    list
        List of crops (left, top)

    """
    rand = np.random.RandomState(seed)
    mask = (np.array(Image.open(mask_file)) == dominating_class).astype(int)
    H, W = mask.shape

    # try a fast approach first. it doesn't guarantee exhausting all feasible crops
    # The fast approach:
    # 1. random pick N pixels of the class, N > num_crop, e.g. 3*num_crop
    # 2. random pick N crops that each contains a pixel picked in previous step
    # 3. hopefully we have at least num_crop crops that meets the threshold
    class_pixels = np.argwhere(mask)[:, ::-1]
    if len(class_pixels) == 0:
        raise RuntimeError("This class is not present in this image.")
    rand_class_pixels = class_pixels[
        rand.choice(len(class_pixels), max(num_crop * 3, 10), replace=False)
    ]
    rand_crops = rand_class_pixels + np.hstack(
        [
            rand.randint(
                low=(-rand_class_pixels[:, 0]).clip(-crop_size[0] + 1, None),
                high=(W - crop_size[0] - rand_class_pixels[:, 0] + 1).clip(None, 1),
                size=(len(rand_class_pixels), 1),
            ),
            rand.randint(
                low=(-rand_class_pixels[:, 1]).clip(-crop_size[1] + 1, None),
                high=(H - crop_size[1] - rand_class_pixels[:, 1] + 1).clip(None, 1),
                size=(len(rand_class_pixels), 1),
            ),
        ]
    )

    qualified_crops = []
    for crop in rand_crops:
        if (tuple(crop) not in qualified_crops) and (
            mask[crop[1] : (crop[1] + crop_size[1]), crop[0] : (crop[0] + crop_size[0])]
        ).sum() >= threshold * (crop_size[0] * crop_size[1]):
            qualified_crops.append(tuple(crop))
        if len(qualified_crops) == num_crop:
            return qualified_crops  # type: ignore

    # if the fast approach cannot find enough feasible crops, try the slow approach
    # The slow approach:
    # 1. check if the threshold is met for every possible crop (by 2D convolution)
    # 2. random pick num_crop feasible crops, if there are enough. raise error if not.
    dominance = correlate1d(
        correlate1d(mask, np.ones(crop_size[1]), axis=0), np.ones(crop_size[0]), axis=1,
    )[
        int(crop_size[1] / 2) : (int(crop_size[1] / 2) + H - crop_size[1]),
        int(crop_size[0] / 2) : (int(crop_size[0] / 2) + W - crop_size[0]),
    ]
    qualified_crops = np.argwhere(
        dominance >= threshold * (crop_size[0] * crop_size[1])
    )[:, ::-1]
    if len(qualified_crops) < num_crop:
        raise RuntimeError(
            f"Found only {len(qualified_crops)} crops "
            f"that are dominated by {threshold*100}%."
        )
    else:
        idx = rand.choice(len(qualified_crops), size=num_crop, replace=False)
        return [tuple(crop) for crop in qualified_crops[idx].tolist()]  # type:ignore
