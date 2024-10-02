# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
import urllib.parse
from typing import List

from direct.types import PathOrString
from direct.utils.io import check_is_valid_url, read_list
import os


def get_filenames_for_datasets_from_config(cfg, files_root: PathOrString, data_root: pathlib.Path):
    """Given a configuration object it returns a list of filenames.

    Parameters
    ----------
    cfg: cfg-object
        cfg object having property lists having the relative paths compared to files root.
    files_root: Union[str, pathlib.Path]
    data_root: pathlib.Path

    Returns
    -------
    list of filenames or None
    """
    if "filenames_lists" not in cfg:
        return None
    lists = cfg.filenames_lists
    return get_filenames_for_datasets(lists, files_root, data_root)


def get_filenames_for_datasets(dataset_name: str,base_path: pathlib.Path, file_names: List[PathOrString]):
    """Given lists of filenames of data points, concatenate these into a large list of full filenames.

    Parameters
    ----------
    lists: List[PathOrString]
    files_root: PathOrString
    data_root: pathlib.Path

    Returns
    -------
    list of filenames or None
    """
    # Build the path, know that files_root can also be a URL
    #is_url = check_is_valid_url(files_root)

    ret = [];

    if dataset_name == 'AHEAD':
        ret.extend(os.path.join(base_path, 'ax', d[:d.rfind('.')] + '_ax' + '.h5') for d in file_names);
        ret.extend(os.path.join(base_path, 'cor', d[:d.rfind('.')] + '_cor' + '.h5') for d in file_names);
        ret.extend(os.path.join(base_path, 'sag',  d[:d.rfind('.')] + '_sag' + '.h5') for d in file_names);
    
    elif dataset_name == 'SKM-TEA':
        ret.extend(os.path.join(base_path, "E1_" + d) for d in file_names);
        ret.extend(os.path.join(base_path, "E2_" + d) for d in file_names);

    else:
        ret = [os.path.join(base_path, f) for f in file_names];

    return ret
