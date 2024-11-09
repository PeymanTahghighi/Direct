# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from torch.utils.data import Dataset

from direct.types import PathOrString
from direct.utils import cast_as_path
from direct.utils.dataset import get_filenames_for_datasets, explicit_zero_padding, parse_fastmri_header
import os
import pickle
from typing import Callable
from tqdm import tqdm
from sklearn.utils import shuffle
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class H5SliceData(Dataset):
    """A PyTorch Dataset class which outputs k-space slices based on the h5 dataformat."""

    def __init__(
        self,
        root: pathlib.Path,
        filenames_filter: Union[List[PathOrString], None] = None,
        filenames_lists: Union[List[PathOrString], None] = None,
        filenames_lists_root: Union[PathOrString, None] = None,
        regex_filter: Optional[str] = None,
        dataset_description: Optional[Dict[PathOrString, Any]] = None,
        metadata: Optional[Dict[PathOrString, Dict]] = None,
        sensitivity_maps: Optional[PathOrString] = None,
        extra_keys: Optional[Tuple] = None,
        pass_attrs: bool = False,
        text_description: Optional[str] = None,
        kspace_context: Optional[int] = None,
        pass_dictionaries: Optional[Dict[str, Dict]] = None,
        pass_h5s: Optional[Dict[str, List]] = None,
        slice_data: Optional[slice] = None,
        data_cache_tranform: Optional[Callable] = None,
        data_type = 'train',
        validation_data_type = 'normal',
        accelerations: int = 4,
        seq = 'all',
        view = 'all'
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        root: pathlib.Path
            Root directory to data.
        filenames_filter: Union[List[PathOrString], None]
            List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
            on the root. If set, will skip searching for files in the root. Default: None.
        filenames_lists: Union[List[PathOrString], None]
            List of paths pointing to `.lst` file(s) that contain file-names in `root` to filter.
            Should be the same as the ones that can be derived from a glob on the root. If this is set,
            this will override the `filenames_filter` option if not None. Defualt: None.
        filenames_lists_root: Union[PathOrString, None]
            Root of `filenames_lists`. Ignored if `filename_lists` is None. Default: None.
        regex_filter: str
            Regular expression filter on the absolute filename. Will be applied after any filenames filter.
        metadata: dict
            If given, this dictionary will be passed to the output transform.
        sensitivity_maps: [pathlib.Path, None]
            Path to sensitivity maps, or None.
        extra_keys: Tuple
            Add extra keys in h5 file to output.
        pass_attrs: bool
            Pass the attributes saved in the h5 file.
        text_description: str
            Description of dataset, can be useful for logging.
        pass_dictionaries: dict
            Pass a dictionary of dictionaries, e.g. if {"name": {"filename_0": val}}, then to `filename_0`s sample dict,
            a key with name `name` and value `val` will be added.
        pass_h5s: dict
            Pass a dictionary of paths. If {"name": path} is given then to the sample of `filename` the same slice
            of path / filename will be added to the sample dictionary and will be asigned key `name`. This can first
            instance be convenient when you want to pass sensitivity maps as well. So for instance:

            >>> pass_h5s = {"sensitivity_map": "/data/sensitivity_maps"}

            will add to each output sample a key `sensitivity_map` with value a numpy array containing the same slice
            of /data/sensitivity_maps/filename.h5 as the one of the original filename filename.h5.
        slice_data : Optional[slice]
            If set, for instance to slice(50,-50) only data within this slide will be added to the dataset. This
            is for instance convenient in the validation set of the public Calgary-Campinas dataset as the first 50
            and last 50 slices are excluded in the evaluation.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.root = pathlib.Path(root)
        self.filenames_filter = filenames_filter
        self.accelerations = accelerations;
        self.metadata = metadata
        self.seq = seq;
        self.view = view;

        self.dataset_description = dataset_description
        self.text_description = text_description

        self.data: List[Tuple] = []

        self.volume_indices: Dict[pathlib.Path, range] = {}
        self.loaded_files = dict();
        self.data_type = data_type;
        self.validation_data_type = validation_data_type
        self.data_postfix = '';
        self.set_data_postfix();

        # If filenames_filter and filenames_lists are given, it will load files in filenames_filter
        # and filenames_lists will be ignored.
        if filenames_filter is None:
            if filenames_lists is not None:
                if filenames_lists_root is None:
                    e = "`filenames_lists` is passed but `filenames_lists_root` is None."
                    self.logger.error(e)
                    raise ValueError(e)
                filenames = get_filenames_for_datasets(
                    lists=filenames_lists, files_root=filenames_lists_root, data_root=root
                )
                self.logger.info("Attempting to load %s filenames from list(s).", len(filenames))
            else:
                self.logger.info("Parsing directory %s for h5 files.", self.root)
                filenames = list(self.root.glob("*.h5"))
        else:
            self.logger.info("Attempting to load %s filenames.", len(filenames_filter))
            filenames = filenames_filter

        filenames = [pathlib.Path(_) for _ in filenames]

        if regex_filter:
            filenames = [_ for _ in filenames if re.match(regex_filter, str(_))]

        if len(filenames) == 0:
            warn = (
                f"Found 0 h5 files in directory {self.root}."
                if not self.text_description
                else f"Found 0 h5 files in directory {self.root} for dataset {self.text_description}."
            )
            self.logger.warning(warn)
        else:
            self.logger.info("Using %s h5 files in %s.", len(filenames), self.root)

        self.kspace_context = kspace_context if kspace_context else 0
        self.pass_h5s = pass_h5s

        self.sensitivity_maps = cast_as_path(sensitivity_maps)
        self.pass_attrs = pass_attrs
        self.extra_keys = extra_keys
        self.pass_dictionaries = pass_dictionaries

        self.ndim = 2 if self.kspace_context == 0 else 3

        self.parse_filenames_data(
            dataset_description, filenames, data_cache_tranform, extra_h5s=pass_h5s, filter_slice=slice_data, data_type = data_type
        )  # Collect information on the image masks_dict.

        if self.text_description:
            self.logger.info("Dataset description: %s.", self.text_description)
            
    def set_data_postfix(self):
        if self.validation_data_type == 'normal':
            self.data_postfix='';
        elif self.validation_data_type == 'equispaced':
            self.data_postfix = '_equispaced';
        elif self.validation_data_type=='inference':
            self.data_postfix ='_inference'
        
        if self.accelerations == 2:
            self.acc_postfix = '_2x';
        else:
            self.acc_postfix = ''
    
    def cache_validation(self, filepaths, transforms, base_root, data_type):
        current_slice_number = 0
        filepaths = shuffle(filepaths, random_state = 42);
        self.logger.info(f'total file path for validation {len(filepaths)}, took 30%: {int(len(filepaths)*0.3)}')
        filepaths = filepaths[:int(len(filepaths)*0.3)];

        for idx, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath);
            filename = filename[:filename.rfind('.')];

            if len(filepaths) < 5 or idx % (len(filepaths) // 5) == 0 or len(filepaths) == (idx + 1):
                self.logger.info(f"Parsing: {(idx + 1) / len(filepaths) * 100:.2f}%.")
            try:
                with h5py.File(filepath, "r") as data:
                    kspace_shape = data["kspace"].shape  # pylint: disable = E1101
                    num_slices = kspace_shape[0]

                    for slice_no in range(num_slices):
                        if os.path.exists(os.path.join(base_root,f"cache_{data_type}", f'{filename}_{slice_no}_cache{self.data_postfix}{self.acc_postfix}.ch')) is True:
                            self.data.append(os.path.join(base_root, f"cache_{data_type}", f'{filename}_{slice_no}_cache{self.data_postfix}{self.acc_postfix}.ch'));
                            continue;
                        
                        kspace, extra_data = self.get_slice_data(data, filepath, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys);

                        sample = {"kspace": kspace, "filename": str(filepath), "slice_no": slice_no}

                        # If the sensitivity maps exist, load these
                        if self.sensitivity_maps:
                            sensitivity_map, _ = self.get_slice_data(self.sensitivity_maps / filepath.name, slice_no)
                            sample["sensitivity_map"] = sensitivity_map

                        sample.update(extra_data)

                        sample.update(parse_fastmri_header(sample, "ismrmrd_header"))
                        if "ismrmrd_header" in sample.keys():
                            del sample["ismrmrd_header"]
                        # Some images have strange behavior, e.g. FLAIR 203.
                        image_shape = sample["kspace"].shape
                        if image_shape[-1] < sample["reconstruction_size"][-2]:  # reconstruction size is (x, y, z)
                            sample["reconstruction_size"] = (image_shape[-1], image_shape[-1], 1)
                        
                        sample["kspace"] = explicit_zero_padding(
                            sample["kspace"], sample["padding_left"], sample["padding_right"]
                            )
                        
                        sample = transforms(sample);

                        with open(os.path.join(base_root,f"cache_{data_type}", f'{filename}_{slice_no}_cache{self.data_postfix}{self.acc_postfix}.ch'), 'wb') as f:
                            pickle.dump(sample, f);
                        self.data.append(os.path.join(base_root, f"cache_{data_type}", f'{filename}_{slice_no}_cache{self.data_postfix}{self.acc_postfix}.ch'));

            except OSError as exc:
                self.logger.warning("%s failed with OSError: %s. Skipping...", filepath, exc)
                continue

            self.volume_indices[filepath] = range(current_slice_number, current_slice_number + num_slices)

            current_slice_number += num_slices

    def cache_validation_inference(self, filepaths, base_root, data_type):
        current_slice_number = 0
        for idx, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath);
            filename = filename[:filename.rfind('.')];

            if len(filepaths) < 5 or idx % (len(filepaths) // 5) == 0 or len(filepaths) == (idx + 1):
                self.logger.info(f"Parsing: {(idx + 1) / len(filepaths) * 100:.2f}%.")
            try:
                with h5py.File(filepath, "r") as data:
                    kspace_shape = data["kspace"].shape  # pylint: disable = E1101
                    num_slices = kspace_shape[0]

                    for slice_no in range(num_slices):
                        if os.path.exists(os.path.join(base_root,f"cache_{data_type}", f'{filename}_{slice_no}_cache_inference.ch')) is True:
                            self.data.append(os.path.join(base_root, f"cache_{data_type}", f'{filename}_{slice_no}_cache_inference.ch'));
                            continue;
                        
                        kspace, extra_data = self.get_slice_data(data, filepath, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys);

                        sample = {"kspace": kspace, "filename": str(filepath), "slice_no": slice_no}

                        # If the sensitivity maps exist, load these
                        if self.sensitivity_maps:
                            sensitivity_map, _ = self.get_slice_data(self.sensitivity_maps / filepath.name, slice_no)
                            sample["sensitivity_map"] = sensitivity_map

                        sample.update(extra_data)

                        with open(os.path.join(base_root,f"cache_{data_type}", f'{filename}_{slice_no}_cache_inference.ch'), 'wb') as f:
                            pickle.dump(sample, f);
                        self.data.append(os.path.join(base_root, f"cache_{data_type}", f'{filename}_{slice_no}_cache_inference.ch'));

            except OSError as exc:
                self.logger.warning("%s failed with OSError: %s. Skipping...", filepath, exc)
                continue

            self.volume_indices[filepath] = range(current_slice_number, current_slice_number + num_slices)

            current_slice_number += num_slices

    def cache_training(self, filepaths, base_root, data_type):
        current_slice_number = 0
        for idx, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath);
            filename = filename[:filename.rfind('.')];

            if len(filepaths) < 5 or idx % (len(filepaths) // 5) == 0 or len(filepaths) == (idx + 1):
                self.logger.info(f"Parsing: {(idx + 1) / len(filepaths) * 100:.2f}%.")
            try:
                with h5py.File(filepath, "r") as data:
                    kspace_shape = data["kspace"].shape  # pylint: disable = E1101
                    num_slices = kspace_shape[0]

                    for slice_no in range(num_slices):
                        if os.path.exists(os.path.join(base_root,f"cache_{data_type}", f'{filename}_{slice_no}_cache.ch')) is True:
                            self.data.append(os.path.join(base_root, f"cache_{data_type}", f'{filename}_{slice_no}_cache.ch'));
                            continue;
                        
                        kspace, extra_data = self.get_slice_data(data, filepath, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys);

                        sample = {"kspace": kspace, "filename": str(filepath), "slice_no": slice_no}

                        # If the sensitivity maps exist, load these
                        if self.sensitivity_maps:
                            sensitivity_map, _ = self.get_slice_data(self.sensitivity_maps / filepath.name, slice_no)
                            sample["sensitivity_map"] = sensitivity_map

                        sample.update(extra_data)

                        with open(os.path.join(base_root,f"cache_{data_type}", f'{filename}_{slice_no}_cache.ch'), 'wb') as f:
                            pickle.dump(sample, f);
                        self.data.append(os.path.join(base_root, f"cache_{data_type}", f'{filename}_{slice_no}_cache.ch'));

                #self.verify_extra_h5_integrity(filepath, kspace_shape, extra_h5s=extra_h5s)  # pylint: disable = E1101

            except OSError as exc:
                self.logger.warning("%s failed with OSError: %s. Skipping...", filepath, exc)
                continue

            self.volume_indices[filepath] = range(current_slice_number, current_slice_number + num_slices)

            current_slice_number += num_slices
          
    def parse_filenames_data(self, dataset_description, filepaths, transforms, extra_h5s=None, filter_slice=None, data_type = 'train'):
        #check if we have already cached this dataset
        if os.path.exists(f"{data_type}_cache_{dataset_description}{self.data_postfix if data_type == 'val' else ''}{self.acc_postfix if data_type == 'val' else ''}.ch") is not False:
            with open(f"{data_type}_cache_{dataset_description}{self.data_postfix if data_type == 'val' else ''}{self.acc_postfix if data_type == 'val' else ''}.ch", 'rb') as f:
                dataset_cache = pickle.load(f);
        else:
            dataset_cache = {};
        

        if dataset_description not in dataset_cache:

            #check if we have cache folder for this dataset, take the root of the first file,
            #cache will be saved beside data root in a folder called cache
            base_root = os.path.dirname(filepaths[0]);
            if os.path.exists(os.path.join(base_root, f'cache_{data_type}')) is False:
                os.makedirs(os.path.join(base_root, f'cache_{data_type}'));

            self.logger.info(f'{dataset_description} does not exists in cache, loading from scratch...')

            if data_type == 'val' and self.validation_data_type != 'inference':
                self.cache_validation(filepaths, transforms, base_root, data_type);
                #done loading files, cache it
                dataset_cache[dataset_description] = [self.data, self.volume_indices]
            elif data_type == 'val' and self.validation_data_type == 'inference': 
                self.cache_validation_inference(filepaths, base_root, data_type);
                #done loading files, cache it
                dataset_cache[dataset_description] = [self.data, self.volume_indices]
            else:
                self.cache_training(filepaths, base_root, data_type)
                #done loading files, cache it
                dataset_cache[dataset_description] = [self.data, self.volume_indices]
        
            with open(f"{data_type}_cache_{dataset_description}{self.data_postfix if data_type == 'val' else ''}{self.acc_postfix if data_type == 'val' else ''}.ch", 'wb') as f:
                pickle.dump(dataset_cache, f);
        
        else:
            self.logger.info(f'{dataset_description} found in cache, loading from cache...')

            self.data, self.volume_indices = dataset_cache[dataset_description];
            
            self.prune_based_on_view_seq(filepaths= filepaths);
            

    def prune_based_on_view_seq(self, filepaths):
        if self.seq != 'all' or self.view != 'all':
            self.logger.info(f'prunning {self.dataset_description}...');
            current_slice_count = 0;
            new_data = [];
            new_volume_indices: Dict[pathlib.Path, range] = {};
            #it means that filesname does not contain whole filenames in the path so we have to prune it here
            for filepath in filepaths:
                rng = self.volume_indices[filepath];
                slices = rng.stop - rng.start;
                d = self.data[rng.start: rng.stop];
                new_data.extend(d);
                new_volume_indices[filepath] = range(current_slice_count, current_slice_count + slices)
                current_slice_count+=slices;
            self.logger.info(f'pruned {self.dataset_description}, old dataset size: {len(self.data)} new dataset size: {len(new_data)}');
            self.data = new_data;
            self.volume_indices = new_volume_indices;
                    

    @staticmethod
    def verify_extra_h5_integrity(image_fn, _, extra_h5s):
        # TODO: This function is not doing much right now, and can be removed or should be refactored to something else
        # TODO: For instance a `direct verify-dataset`?
        if not extra_h5s:
            return

        for key in extra_h5s:
            h5_key, path = extra_h5s[key]
            extra_fn = path / image_fn.name
            try:
                with h5py.File(extra_fn, "r") as file:
                    _ = file[h5_key].shape
            except (OSError, TypeError) as exc:
                raise ValueError(f"Reading of {extra_fn} for key {h5_key} failed: {exc}.") from exc

            # TODO: This is not so trivial to do it this way, as the shape depends on context
            # if image_shape != shape:
            #     raise ValueError(f"{extra_fn} and {image_fn} has different shape. "
            #                      f"Got {shape} and {image_shape}")

    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        with open(self.data[idx], 'rb') as f:
            sample = pickle.load(f);
        #print(f'loading {self.data[idx]} took : {time.time() - t0}');
        #self.logger.info(f'loading {self.data[idx]} took : {time.time() - t0} size: {os.path.getsize(self.data[idx]) / (1024 * 1024)}')
        # filename, slice_no = self.data[idx]
        # filename = pathlib.Path(filename)
        # metadata = None if not self.metadata else self.metadata[filename.name]

        # kspace, extra_data = self.get_slice_data(
        #     filename, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys
        # )

        # if kspace.ndim == 2:  # Singlecoil data does not always have coils at the first axis.
        #     kspace = kspace[np.newaxis, ...]

        # # TODO: Write a custom collate function which disables batching for certain keys
        

        # # If the sensitivity maps exist, load these
        # if self.sensitivity_maps:
        #     sensitivity_map, _ = self.get_slice_data(self.sensitivity_maps / filename.name, slice_no)
        #     sample["sensitivity_map"] = sensitivity_map

        # if metadata is not None:
        #     sample["metadata"] = metadata

        # 

        # if self.pass_dictionaries:
        #     for key in self.pass_dictionaries:
        #         if key in sample:
        #             raise ValueError(f"Trying to add key {key} to sample dict, but this key already exists.")
        #         sample[key] = self.pass_dictionaries[key][filename.name]

        # if self.pass_h5s:
        #     for key, (h5_key, path) in self.pass_h5s.items():
        #         curr_slice, _ = self.get_slice_data(path / filename.name, slice_no, key=h5_key)
        #         if key in sample:
        #             raise ValueError(f"Trying to add key {key} to sample dict, but this key already exists.")
        #         sample[key] = curr_slice

        return sample

    def get_slice_data(self, data, filename, slice_no, key="kspace", pass_attrs=False, extra_keys=None):
        extra_data = {}
        

        if self.kspace_context == 0:
            curr_data = data[key][slice_no]
        else:
            # This can be useful for getting stacks of slices.
            num_slices = self.get_num_slices(filename)
            curr_data = data[key][
                max(0, slice_no - self.kspace_context) : min(slice_no + self.kspace_context + 1, num_slices),
            ]
            curr_shape = curr_data.shape
            if curr_shape[0] < num_slices - 1:
                if slice_no - self.kspace_context < 0:
                    new_shape = list(curr_shape).copy()
                    new_shape[0] = self.kspace_context - slice_no
                    curr_data = np.concatenate(
                        [np.zeros(new_shape, dtype=curr_data.dtype), curr_data],
                        axis=0,
                    )
                if self.kspace_context + slice_no > num_slices - 1:
                    new_shape = list(curr_shape).copy()
                    new_shape[0] = slice_no + self.kspace_context - num_slices + 1
                    curr_data = np.concatenate(
                        [curr_data, np.zeros(new_shape, dtype=curr_data.dtype)],
                        axis=0,
                    )
            # Move the depth axis to the second spot.
            curr_data = np.swapaxes(curr_data, 0, 1)

        if pass_attrs:
            extra_data["attrs"] = dict(data.attrs)

        if extra_keys:
            for extra_key in self.extra_keys:
                if extra_key == "attrs":
                    raise ValueError("attrs need to be passed by setting `pass_attrs = True`.")
                if extra_key in data.keys():
                    extra_data[extra_key] = data[extra_key][()]
        return curr_data, extra_data

    def get_num_slices(self, filename):
        num_slices = self.volume_indices[filename].stop - self.volume_indices[filename].start
        return num_slices
