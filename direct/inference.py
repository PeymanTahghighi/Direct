# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import sys
from functools import partial
from typing import Callable, DefaultDict, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig

from direct.utils.dataset import get_filenames_for_datasets
from direct.data.datasets import build_dataset_from_input
from direct.data.mri_transforms import build_mri_transforms
from direct.environment import setup_inference_environment
from direct.types import FileOrUrl, PathOrString
from direct.utils import chunks, dict_flatten, remove_keys
from direct.utils.io import read_list
from direct.utils.writers import write_output_to_h5
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def setup_inference_save_to_h5(
    get_inference_settings: Callable,
    run_name: str,
    data_sheet: str,
    data_root: Union[PathOrString, None],
    base_directory: PathOrString,
    output_directory: PathOrString,
    filenames_filter: Union[List[PathOrString], None],
    checkpoint: FileOrUrl,
    device: str,
    num_workers: int,
    machine_rank: int,
    cfg_file: Union[PathOrString, None] = None,
    process_per_chunk: Optional[int] = None,
    mixed_precision: bool = False,
    debug: bool = False,
    is_validation: bool = False,
    
) -> None:
    """This function contains most of the logic in DIRECT required to launch a multi-gpu / multi-node inference process.

    It saves predictions as `.h5` files.

    Parameters
    ----------
    get_inference_settings: Callable
        Callable object to create inference dataset and environment.
    run_name: str
        Experiment run name. Can be an empty string.
    data_root: Union[PathOrString, None]
        Path of the directory of the data if applicable for dataset. Can be None.
    base_directory: PathOrString
        Path to directory where where inference logs will be stored. If `run_name` is not an empty string,
        `base_directory / run_name` will be used.
    output_directory: PathOrString
        Path to directory where output data will be saved.
    filenames_filter: Union[List[PathOrString], None]
        List of filenames to include in the dataset (if applicable). Can be None.
    checkpoint: FileOrUrl
        Checkpoint to a model. This can be a path to a local file or an URL.
    device: str
        Device name.
    num_workers: int
        Number of workers.
    machine_rank: int
        Machine rank.
    cfg_file: Union[PathOrString, None]
        Path to configuration file. If None, will search in `base_directory`.
    process_per_chunk: int
        Processes per chunk number.
    mixed_precision: bool
        If True, mixed precision will be allowed. Default: False.
    debug: bool
        If True, debug information will be displayed. Default: False.
    is_validation: bool
        If True, will use settings (e.g. `batch_size` & `crop`) of `validation` in config.
        Otherwise it will use `inference` settings. Default: False.

    Returns
    -------
    None
    """
    env = setup_inference_environment(
        run_name, base_directory, device, machine_rank, mixed_precision, cfg_file, debug=debug
    )

    dataset_cfg, transforms = get_inference_settings(env)

    torch.backends.cudnn.benchmark = True
    logger.info(f"Predicting dataset and saving in: {output_directory}.")

    datasets = []
    for idx, (transform, cfg) in enumerate(zip(transforms, dataset_cfg)):
        if cfg.transforms.masking is None:  # type: ignore
            logger.info(
                "Masking function set to None for %s.",
                dataset_config.text_description,  # type: ignore
            )

        dataset_args = {"transforms": transform, "dataset_config": cfg}
        
        if data_sheet is not None:
            xls = pd.ExcelFile(data_sheet);
            sheet_name = dataset_args['dataset_config']['sheet_name']
            df = pd.read_excel(xls, sheet_name);
            names = [df.loc[l, 'Name'] for l in np.where(df['New subsets'] == cfg.set_type)[0]]
            data_root = cfg['base_path']
            dataset_args.update({"data_root": data_root})
            filenames_filter = get_filenames_for_datasets(cfg['sheet_name'], data_root, names)
            dataset_args.update({"filenames_filter": filenames_filter})   
        dataset_args.update({'data_type': cfg.set_type});
        dataset_args.update({'validation_data_type': 'inference'});
        dataset = build_dataset_from_input(**dataset_args)

        datasets.append(dataset)
        logger.info(
            "Data size for %s (%s/%s): %s.",
            cfg.text_description,  # type: ignore
            idx + 1,
            len(dataset_cfg),
            len(dataset),
        )


        if is_validation:
            batch_size, crop = env.cfg.validation.batch_size, env.cfg.validation.crop  # type: ignore
        else:
            batch_size, crop = env.cfg.inference.batch_size, env.cfg.inference.crop  # type: ignore

        
        output = inference_on_environment(
            env=env,
            dataset=dataset,
            experiment_path=base_directory / run_name,
            checkpoint=checkpoint,
            num_workers=num_workers,
            batch_size=batch_size,
            crop=crop,
        )

        # Perhaps aggregation to the main process would be most optimal here before writing.
        # The current way this write the volumes for each process.
    write_output_to_h5(
            output,
            output_directory,
            output_key="reconstruction",
        )


def build_inference_transforms(env, mask_func: Callable, dataset_cfg: DictConfig) -> Callable:
    """Builds inference transforms."""
    partial_build_mri_transforms = partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=mask_func,
    )
    transforms = partial_build_mri_transforms(**dict_flatten(remove_keys(dataset_cfg.transforms, "masking")))
    return transforms


def inference_on_environment(
    env,
    dataset,
    experiment_path: PathOrString,
    checkpoint: FileOrUrl,
    num_workers: int = 0,
    batch_size: int = 1,
    crop: Optional[str] = None,
) -> Union[Dict, DefaultDict]:
    """Performs inference on environment.

    Parameters
    ----------
    env: Environment.
    data_root: Union[PathOrString, None]
        Path of the directory of the data if applicable for dataset. Can be None.
    dataset_cfg: DictConfig
        Configuration containing inference dataset settings.
    transforms: Callable
        Dataset transformations object.
    experiment_path: PathOrString
        Path to directory where where inference logs will be stored.
    checkpoint: FileOrUrl
        Checkpoint to a model. This can be a path to a local file or an URL.
    num_workers: int
        Number of workers.
    filenames_filter: Union[List[PathOrString], None]
        List of filenames to include in the dataset (if applicable). Can be None. Default: None.
    batch_size: int
        Inference batch size. Default: 1.
    crop: Optional[str]
        Inference crop type. Can be `header` or None. Default: None.

    Returns
    -------
    output: Union[Dict, DefaultDict]
    """
    logger.info(f"Inference data size: {len(dataset)}.")

    # Run prediction
    output = env.engine.predict(
        dataset,
        experiment_path,
        checkpoint=checkpoint,
        batch_size=batch_size,
        crop=crop,
        num_workers = env.cfg.inference.num_workers,
        prefetch_factor = env.cfg.inference.prefetch_factor,
    )
    return output
