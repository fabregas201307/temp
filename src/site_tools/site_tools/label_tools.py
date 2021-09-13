import json
import os
from io import BytesIO
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from PIL import Image

from .core_tools import parallel


def _export_semantic_segmentation_from_one_image(
    label, class_to_idx, mask_dir, image_dir, thicken_polyline, verbose
):
    try:
        orig_image_path = label["External ID"]
        if verbose:
            print(orig_image_path)

        if not label["Label"]:  # nothing labeled
            return

        image_url = label["Labeled Data"]
        response = requests.get(image_url, allow_redirects=True)
        image_file_name = label["External ID"].replace("/", "_-_")
        with open(  # type:ignore
            os.path.join(image_dir, image_file_name), "wb"
        ) as f:
            f.write(response.content)  # type:ignore

        img = np.array(Image.open(os.path.join(image_dir, image_file_name)))
        H, W, _ = img.shape

        multiclass_mask = 255 * np.ones((H, W))
        reserved_for_polylines = np.zeros((H, W))
        for obj in label["Label"]["objects"]:
            if obj["title"] not in class_to_idx.keys():
                continue
            if "line" in obj.keys():
                is_polyline_obj = True
            else:
                is_polyline_obj = False
            class_id = class_to_idx[obj["title"]]
            mask_url = obj["instanceURI"]
            response = requests.get(mask_url)
            try:
                mask = np.array(Image.open(BytesIO(response.content)).convert("L"))
                if mask.shape != (H, W):
                    print("Orig image: ", label["External ID"])
                    print("Rescaling mask...")
                    mask = np.array(Image.fromarray(mask).resize((W, H)))
                mask = (mask / 255).astype(bool)
                if is_polyline_obj:
                    mask = _expand_segmentation(mask, thicken_polyline)
                multiclass_mask[
                    mask & (1 - reserved_for_polylines).astype(bool)
                ] = class_id
                if is_polyline_obj:
                    reserved_for_polylines[mask] = 1
            except OSError:
                # This catches errors if the url doesn't contain an image (likely
                # meaning it was deleted from labelbox)
                print("Orig image: ", label["External ID"])
                print("Error reading url: ", mask_url)
                print("Response: ", response.content)

        multiclass_mask = multiclass_mask.astype(np.uint8)
        mask_img = Image.fromarray(multiclass_mask)
        mask_file_name = f"{label['External ID'].replace('/', '_-_')[:-4]}.png"
        mask_img.save(os.path.join(mask_dir, mask_file_name))

    except Exception as e:
        print("Orig image: ", label["External ID"])
        print(e)


def export_semantic_segmentation(  # noqa: C901
    json_file: str,
    mask_dir: str,
    image_dir: Optional[str] = None,
    ignore_classes: Optional[List[str]] = None,
    class_alias: Optional[Dict[str, Union[str, List[str]]]] = None,
    thicken_polyline: Optional[int] = 0,
    max_labels: Optional[int] = None,
    verbose: Optional[bool] = True,
) -> None:
    """
    Export pixel-wise semantic segmentation from JSON exported by LabelBox.

    The multi-class mask of every image is saved as grayscale PNG file, where a
    grayscale level correponds a class, and grayscale 255 corresponds no class. The mask
    images are saved in the output directory following the with the same path and file
    name as the original image file but with '/' replaced by '_-_'. For example, the
    mask of image 'folder/subfolder/image.JPG' will be saved as 'mask_dir_-_folder_-_
    subfolder_-_image.png'. A JSON file "classes.json" that maps all classes to the
    corresponding grayscale is also saved in output directory.

    The original images will be downloaded to `image_dir`.

    Note this function is for semantic segmentation. All instance of the same class will
    be combined and information of instances will be lost in the output. All bounding
    boxes will be ignored.

    Parameters
    ----------
    json_file : str
        The path to the JSON file exported by LabelBox

    mask_dir : str
        The directory to save segmentation masks.

    image_dir : str
        The directory to save the original images corresponding to the exported masks.

    ignore_classes : list, optional
        A list of segmentation classes that should be ignored during exporting. If not
        given, all segmentation classes in the JSON file will be exported. Default: None

    class_alias : dict, optional
        A dictionary maps class aliases. For example, {"A": ["a", "alpha"], "B": "b"}
        means classes "a" and "alpha" will be treated as "A", and class "b" will be
        treated as "B". Default: None.

    thicken_polyline : int, optional
        Degree of expansion for pixel-wise segmentation defined by a polyline.
        Default: 0, i.e. only treat the pixels passed by the polyline as the class.

    max_labels : int, optional
        Maximal number of labels to export. If not given, export all in the JSON file.
        Default: None.

    verbose : bool, optional
        Print level. Default: True.

    """

    with open(json_file, "r") as f:
        labels = json.load(f)

    if ignore_classes is None:
        ignore_classes = []

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    if (image_dir is not None) and (not os.path.exists(image_dir)):
        os.makedirs(image_dir)

    if class_alias is not None:
        class_alias_reverse = dict()
        for key, value in class_alias.items():
            if isinstance(value, str):
                class_alias_reverse[value] = key
            else:
                for alias in value:
                    class_alias_reverse[alias] = key
    else:
        class_alias_reverse = dict()

    class_to_idx = dict()  # type: Dict[str, int]
    for label in labels:
        if not label["Label"]:
            continue
        for obj in label["Label"]["objects"]:
            if obj["title"] in class_to_idx.keys():
                continue
            if "bbox" in obj.keys():
                continue
            if (obj["title"] in ignore_classes) or (
                (obj["title"] in class_alias_reverse.keys())
                and (class_alias_reverse[obj["title"]] in ignore_classes)
            ):
                continue
            if obj["title"] not in class_alias_reverse.keys():
                class_to_idx[obj["title"]] = len(class_to_idx)
            else:
                if class_alias_reverse[obj["title"]] in class_to_idx.keys():
                    class_to_idx[obj["title"]] = class_to_idx[
                        class_alias_reverse[obj["title"]]
                    ]
                else:
                    class_to_idx[class_alias_reverse[obj["title"]]] = class_to_idx[
                        obj["title"]
                    ]
                    class_to_idx[obj["title"]] = len(class_to_idx)

    with open(os.path.join(mask_dir, "classes.json"), "w") as f:
        json.dump({key: value for key, value in class_to_idx.items()}, f, indent=2)

    parallel(
        _export_semantic_segmentation_from_one_image,
        items=labels if (max_labels is None) else labels[:max_labels],
        class_to_idx=class_to_idx,
        mask_dir=mask_dir,
        image_dir=image_dir,
        verbose=verbose,
        thicken_polyline=thicken_polyline,
    )


def _expand_segmentation(mask, expand):
    """
    mask : binary numpy array (H, W)
    expand : int, number of pixels to expand
    """
    if expand == 0:
        return mask.copy()
    if expand == 1:
        H, W = mask.shape
        idx = np.argwhere(mask)
        up_shift_idx = idx.copy()
        up_shift_idx[:, 0] -= 1
        up_shift_idx = up_shift_idx[up_shift_idx.min(axis=1) >= 0]
        down_shift_idx = idx.copy()
        down_shift_idx[:, 0] += 1
        down_shift_idx = down_shift_idx[down_shift_idx.max(axis=1) < H]
        left_shift_idx = idx.copy()
        left_shift_idx[:, 1] -= 1
        left_shift_idx = left_shift_idx[left_shift_idx.min(axis=1) >= 0]
        right_shift_idx = idx.copy()
        right_shift_idx[:, 1] += 1
        right_shift_idx = right_shift_idx[right_shift_idx.max(axis=1) < W]
        expanded = np.zeros((H, W))
        expanded[idx[:, 0], idx[:, 1]] = 1
        expanded[up_shift_idx[:, 0], up_shift_idx[:, 1]] = 1
        expanded[down_shift_idx[:, 0], down_shift_idx[:, 1]] = 1
        expanded[left_shift_idx[:, 0], left_shift_idx[:, 1]] = 1
        expanded[right_shift_idx[:, 0], right_shift_idx[:, 1]] = 1
        return expanded.astype(bool)
    if expand > 1:
        return _expand_segmentation(
            _expand_segmentation(mask, expand=1), expand=expand - 1
        )
    raise ValueError(f"Expand {expand} is invalid.")
