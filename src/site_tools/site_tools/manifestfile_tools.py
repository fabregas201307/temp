import json
from pathlib import Path
from typing import Iterator

from sklearn.model_selection import train_test_split

from .aws_tools import get_boto3_session, get_matching_s3_keys


def create_aug_manifest_v2(
    labelbox_images: Iterator[str],
    bucket: str,
    manifest_file_path: str = "data/",
    verbose: bool = False,
) -> None:
    """Uploads 3 manigest files in s3 bucket at {manifest_file_path}
       images and image annotation should follow following naming
       images folder = {image}
       images annotation folder = {image}_annotation
       For Example
       images = s3://data/train
       annotations = s3://data/train_annotation

    Args:
        labelbox_images (generator): list of files in s3 train folder
        bucket (string): [bucket name]
        manifest_file_path (str, optional): location for storing manifest file. Defaults
        to "data/".
        verbose (bool): Print outputs to stdout (default is False)

    """

    #   getting session
    session = get_boto3_session(bucket)
    # s3 = session.resource("s3")
    s3_client = session.client("s3")

    imgs_list = [f for f in labelbox_images]

    # TODO: refactor to allow specifying train/valid/test split. First split on
    # train/test+valid and then split on test/valid.
    train_imgs, test_imgs = train_test_split(
        imgs_list, train_size=0.9, test_size=0.1, shuffle=True, random_state=42
    )

    train_imgs, val_imgs = train_test_split(
        train_imgs, train_size=0.9, test_size=0.1, shuffle=True, random_state=42
    )
    #   dict with splited list of images
    imgs_splits = {
        "train_imgs": train_imgs,
        "test_imgs": test_imgs,
        "val_imgs": val_imgs,
    }
    for key, imgs in imgs_splits.items():
        print(key, len(imgs))

        manifest_file = f"{manifest_file_path}/manifest_file_{key}.json"
        obj = ""
        print(manifest_file)
        for f in imgs:
            # getting image path plus file name minus the extension
            p = Path(f)

            file_nam_wo_ext = p.stem

            base_loc = f.split("/")[:-2]
            base_loc = "/".join(base_loc)

            # img_path = f"s3://{bucket}/{base_loc}/images/{file_nam_wo_ext}.jpg"
            # adding temporary fix till we have all the images following .jpg extension.
            file_name = p.name
            img_path = f"s3://{bucket}/{base_loc}/images/{file_name}"
            mask_path = f"s3://{bucket}/{base_loc}/annotations/{file_nam_wo_ext}.png"
            img_mask = {
                "source-ref": img_path,
                "annotation-ref": mask_path,
            }
            # Serializing json
            json_data = json.dumps(img_mask)
            # appending
            obj = obj + json_data + "\n"
        # pushing files to s3
        # s3object = s3.Object(bucket, manifest_file)
        # response = s3object.put(Body=json.dumps(json_object).encode("UTF-8"))
        response = s3_client.put_object(Body=obj, Bucket=bucket, Key=manifest_file)
        if verbose:
            print(response)


def get_keys_and_create_manifest(
    bucket: str, prefix: str, train_path: str = None, manifest_file_path: str = None
) -> None:
    """Get labelbox image files in a specific bucket and create manifest files.

    This creates three manifest files for training/test/validation.

    Parameters
    ----------
    bucket (string): [bucket name]
    prefix (string): prefix path to images and annotations folder
    train_path: prefix path to images folder. Defaults to `{prefix}/images`.
    manifest_file_path (str, optional): location for storing manifest file.
        Defaults to `{prefix}/manifests`.
    """
    if train_path is None:
        train_path = f"{prefix}/images"
    if manifest_file_path is None:
        manifest_file_path = f"{prefix}/manifests"

    labelbox_images_lst = get_matching_s3_keys(
        bucket=bucket, prefix=train_path, suffix="jpg"
    )

    create_aug_manifest_v2(
        labelbox_images_lst, bucket, manifest_file_path=manifest_file_path
    )
