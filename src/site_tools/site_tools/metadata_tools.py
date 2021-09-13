from pathlib import Path
from typing import Union

import exiftool
import pandas as pd

from .image_tools import get_image_files, verify_images


def extract_metadata(filepath: Union[str, Path], verify: bool = False) -> pd.DataFrame:
    """Extracts metadata from all files recursively in `filepath`. Optionally,
    `verify` all images can be loaded and are in fact image files.
    """
    fns = get_image_files(filepath)
    bad = []
    if verify:
        bad = verify_images(fns)
    fns = [str(fn) for fn in fns if fn not in set(bad)]

    with exiftool.ExifTool() as et:
        metadata = et.get_metadata_batch(fns)

    df = pd.DataFrame(metadata)
    return fixup_metadata(df, filepath)


def fixup_metadata(df: pd.DataFrame, filepath: Union[str, Path]) -> pd.DataFrame:
    """Fix the directory path to be local to the top of the data folder. Perform
    various other fixes on the metadata and remove irrelevant columns.
    """
    df["SourceFile"] = df["SourceFile"].str.replace(str(filepath) + "/", "")
    df["File:Directory"] = df["File:Directory"].str.replace(str(filepath) + "/", "")
    df["File:Directory1"] = (
        df["File:Directory"]
        .str.findall(r"(\w+)/")
        .apply(lambda x: "".join(map(str, x)))
    )
    df["File:Directory2"] = (
        df["File:Directory"]
        .str.findall(r"\w+/(\w+)")
        .apply(lambda x: "".join(map(str, x)))
    )
    df["File:Directory3"] = (
        df["File:Directory"]
        .str.findall(r"\w+/\w+/(\w+)")
        .apply(lambda x: "".join(map(str, x)))
    )

    drop_cols = [
        "File:FileModifyDate",
        "File:FileAccessDate",
        "File:FileInodeChangeDate",
        "File:FilePermissions",
    ]
    df.drop(drop_cols, axis=1, inplace=True)

    df.set_index("SourceFile", inplace=True)

    return df
