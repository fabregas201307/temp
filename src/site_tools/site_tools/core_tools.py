import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, Optional, Union


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


def num_cpus() -> Optional[int]:
    """Get number of cpus."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def parallel(
    func: Callable, items: list, *args, n_workers: int = None, **kwargs
) -> Iterator:
    """Applies `func` in parallel to `items`, using `n_workers`.

    Parameters
    ----------
    func: Function to be performed in parallel.
    items: Items to perform the parallel computation on.
    n_workers: Number of workers to run in parallel.
        None (default) sets it to the min of either the number of cpu cores or 16.

    Returns
    -------
    An iterator of the completed computation on the `items`.
    """
    """"""
    if n_workers is None:
        n_workers = min(16, num_cpus())
    func = partial(func, *args, **kwargs)
    with ProcessPoolExecutor(n_workers) as ex:
        return ex.map(func, items)
