# coding=utf-8
# Copyright (c) DIRECT Contributors
import pathlib
import urllib.parse
from typing import List

from direct.types import PathOrString
from direct.utils.io import check_is_valid_url, read_list
import os
import numpy as np
import xml.etree.ElementTree as etree  # nosec
import pandas as pd
from glob import glob
from typing import Sequence

def _et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.

    From:
    https://github.com/facebookresearch/fastMRI/blob/13560d2f198cc72f06e01675e9ecee509ce5639a/fastmri/data/mri_data.py#L23

    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

def parse_fastmri_header(data: dict, key: str) -> dict:
    # Borrowed from: https://github.com/facebookresearch/\
    # fastMRI/blob/13560d2f198cc72f06e01675e9ecee509ce5639a/fastmri/data/mri_data.py#L23
    if key in data.keys():
        xml_header = data[key];
        et_root = etree.fromstring(xml_header)  # nosec

        encodings = ["encoding", "encodedSpace", "matrixSize"]
        encoding_size = (
            int(_et_query(et_root, encodings + ["x"])),
            int(_et_query(et_root, encodings + ["y"])),
            int(_et_query(et_root, encodings + ["z"])),
        )
        reconstructions = ["encoding", "reconSpace", "matrixSize"]
        reconstruction_size = (
            int(_et_query(et_root, reconstructions + ["x"])),
            int(_et_query(et_root, reconstructions + ["y"])),
            int(_et_query(et_root, reconstructions + ["z"])),
        )

        limits = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        encoding_limits_center = int(_et_query(et_root, limits + ["center"]))
        encoding_limits_max = int(_et_query(et_root, limits + ["maximum"])) + 1

        padding_left = encoding_size[1] // 2 - encoding_limits_center
        padding_right = padding_left + encoding_limits_max

        
    else:
        #sample lack ismrmrd_header so we create the metadata with zero data
        padding_left = 0
        padding_right = 0
        encoding_size = (0, 0, 0)
        reconstruction_size = (0, 0, 0)

    metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": encoding_size,
            "reconstruction_size": reconstruction_size,
        }
    return metadata

def explicit_zero_padding(kspace: np.ndarray, padding_left: int, padding_right: int) -> np.ndarray:
    if padding_left > 0:
        kspace[..., 0:padding_left] = 0 + 0 * 1j
    if padding_right > 0:
        kspace[..., padding_right:] = 0 + 0 * 1j

    return kspace


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


def get_filenames_for_datasets(dataset_name: str, 
                               dataset_disc: str,
                               base_path, data_frame: pd, 
                               data_type = 'train',
                               seq: str = 'all', 
                               view: str = 'all'):
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
    ret = [];
    if isinstance(base_path, str):
        if dataset_name == 'AHEAD':
            file_names = [data_frame.loc[l, 'Name'] for l in np.where(data_frame['New subsets'] == data_type)[0]]
            if view == 'Ax' or view == 'all':
                ret.extend(os.path.join(base_path, 'ax', d[:d.rfind('.')] + '_ax' + '.h5') for d in file_names);
            
            if view == 'Cor' or view == 'all':
                ret.extend(os.path.join(base_path, 'cor', d[:d.rfind('.')] + '_cor' + '.h5') for d in file_names);
            
            if view == 'Sag' or view == 'all':
                ret.extend(os.path.join(base_path, 'sag',  d[:d.rfind('.')] + '_sag' + '.h5') for d in file_names);
        
        elif dataset_name == 'SKM-TEA':
            file_names = [data_frame.loc[l, 'Name'] for l in np.where(data_frame['New subsets'] == data_type)[0]]
            ret.extend(os.path.join(base_path, "E1_" + d) for d in file_names);
            ret.extend(os.path.join(base_path, "E2_" + d) for d in file_names);
        elif dataset_name == 'Phantom':
            file_names = glob(os.path.join(base_path, '*.h5'));
            ret.extend(file_names);
        
        else:
            if dataset_name != None:
                file_names = ([data_frame.loc[l, 'Name'] for l in np.where(data_frame['New subsets'] == data_type)[0]])
                if seq != 'all':
                    file_names_seq = ([data_frame.loc[l, 'Name'] for l in np.where(data_frame['seq'] == seq)[0]])
                    file_names = np.intersect1d(file_names, file_names_seq);
                if view != 'all':
                    file_names_view = ([data_frame.loc[l, 'Name'] for l in np.where(data_frame['view'] == view)[0]])
                    file_names = np.intersect1d(file_names, file_names_view);
                
                ret = [os.path.join(base_path, f) for f in file_names];
            else:
                if dataset_disc == 'Special':
                    ret.extend(glob(os.path.join(base_path,'*.h5')))
                elif dataset_disc == 'Synthetic':
                    ret.extend(glob(os.path.join(base_path,'*.h5')))
    else:
        file_names = ([data_frame.loc[l, 'Name'] for l in np.where(data_frame['New subsets'] == data_type)[0]])
        for i in range(len(base_path)):
            temp_list = [];

            if dataset_name == 'AHEAD':
                if view == 'Ax' or view == 'all':
                    temp_list.extend(os.path.join(base_path[i], d[:d.rfind('.')] + '_ax' + '.h5') for d in file_names);
            
                if view == 'Cor' or view == 'all':
                    temp_list.extend(os.path.join(base_path[i], d[:d.rfind('.')] + '_cor' + '.h5') for d in file_names);
                
                if view == 'Sag' or view == 'all':
                    temp_list.extend(os.path.join(base_path[i],  d[:d.rfind('.')] + '_sag' + '.h5') for d in file_names);
            
            elif dataset_name == 'SKM-TEA':
                temp_list.extend(os.path.join(base_path[i], "E1_" + d) for d in file_names);
                temp_list.extend(os.path.join(base_path[i], "E2_" + d) for d in file_names);

            else:
                if seq != 'all':
                    file_names_seq = ([data_frame.loc[l, 'Name'] for l in np.where(data_frame['seq'] == seq)[0]])
                    file_names = np.intersect1d(file_names, file_names_seq);
                if view != 'all':
                    file_names_view = ([data_frame.loc[l, 'Name'] for l in np.where(data_frame['view'] == view)[0]])
                    file_names = np.intersect1d(file_names, file_names_view);
                    
                temp_list = [os.path.join(base_path[i], f) for f in file_names];
            ret.append(temp_list);
    return ret
