# coding=utf-8
# Copyright (c) DIRECT Contributors
import argparse
import functools
import logging
import os
import pathlib
import sys
import urllib.parse
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

from direct.cli.utils import check_train_val
from direct.common.subsample import build_masking_function
from direct.data.datasets import build_dataset_from_input
from direct.data.lr_scheduler import WarmupMultiStepLR
from direct.data.mri_transforms import build_mri_transforms
from direct.environment import setup_training_environment
from direct.launch import launch
from direct.types import PathOrString
from direct.utils import dict_flatten, remove_keys, set_all_seeds, str_to_class
from direct.utils.dataset import get_filenames_for_datasets
from direct.utils.io import check_is_valid_url, read_json
from sklearn.utils import shuffle
import pandas as pd

logger = logging.getLogger(__name__)


def parse_noise_dict(noise_dict: dict, percentile: float = 1.0, multiplier: float = 1.0):
    logger.info("Parsing noise dictionary...")
    output: Dict = defaultdict(dict)
    for filename in noise_dict:
        data_per_volume = noise_dict[filename]
        for slice_no in data_per_volume:
            curr_data = data_per_volume[slice_no]
            if percentile != 1.0:
                lower_clip = np.percentile(curr_data, 100 * (1 - percentile))
                upper_clip = np.percentile(curr_data, 100 * percentile)
                curr_data = np.clip(curr_data, lower_clip, upper_clip)

            output[filename][int(slice_no)] = (
                curr_data * multiplier
            ) ** 2  # np.asarray(curr_data) * multiplier# (np.clip(curr_data, lower_clip, upper_clip) * multiplier) ** 2

    return output


def get_root_of_file(filename: PathOrString):
    """Get the root directory of the file or URL to file.

    Examples
    --------
    >>> get_root_of_file('/mnt/archive/data.txt')
    >>> /mnt/archive
    >>> get_root_of_file('https://aiforoncology.nl/people')
    >>> https://aiforoncology.nl/

    Parameters
    ----------
    filename: pathlib.Path or str

    Returns
    -------
    pathlib.Path or str
    """
    if check_is_valid_url(str(filename)):
        filename = urllib.parse.urljoin(str(filename), ".")
    else:
        filename = pathlib.Path(filename).parents[0]

    return filename


def build_transforms_from_environment(env, dataset_config: DictConfig) -> Callable:
    masking = dataset_config.transforms.masking  # Masking func can be None
    mask_func = None if masking is None else build_masking_function(**masking)
    mri_transforms_func = functools.partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=mask_func,
        to_tensor = True,
        estimate_sensitivity_maps = False
    )

    mri_data_cache_transforms_func = functools.partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=None,
        to_tensor = False,
        delete_acs_mask = False,
        delete_kspace = False,
        apply_mask = False,
        compute_scaling_factor = False,
        estimate_sensitivity_maps = False
    )

    return mri_transforms_func(**dict_flatten(dict(remove_keys(dataset_config.transforms, "masking")))), mri_data_cache_transforms_func(**dict_flatten(dict(remove_keys(remove_keys(dataset_config.transforms, "sensitivity_map_estimation"), "masking"))))  # type: ignore


def build_training_datasets_from_environment(
    env,
    datasets_config: List[DictConfig],
    lists_root: Optional[PathOrString] = None,
    data_sheet: Optional[PathOrString] = None,
    initial_images: Optional[Union[List[pathlib.Path], None]] = None,
    initial_kspaces: Optional[Union[List[pathlib.Path], None]] = None,
    pass_text_description: bool = True,
    pass_dictionaries: Optional[Dict[str, Dict]] = None,
    data_type: str = 'train'
):
    datasets = []
    for idx, dataset_config in enumerate(datasets_config):
        if pass_text_description:
            if not "text_description" in dataset_config:
                dataset_config.text_description = f"ds{idx}" if len(datasets_config) > 1 else None
        else:
            dataset_config.text_description = None
        if dataset_config.transforms.masking is None:  # type: ignore
            logger.info(
                "Masking function set to None for %s.",
                dataset_config.text_description,  # type: ignore
            )
        train_transforms, data_cache_tranform = build_transforms_from_environment(env, dataset_config)
        dataset_args = {"transforms": train_transforms, "dataset_config": dataset_config}
        if initial_images is not None:
            dataset_args.update({"initial_images": initial_images})
        if initial_kspaces is not None:
            dataset_args.update({"initial_kspaces": initial_kspaces})
        if data_sheet is not None:
            xls = pd.ExcelFile(data_sheet);
            sheet_name = dataset_args['dataset_config']['sheet_name']
            df = pd.read_excel(xls, sheet_name);
            names = [df.loc[l, 'Name'] for l in np.where(df['New subsets'] == data_type)[0]]
            data_root = dataset_config['base_path']
            dataset_args.update({"data_root": data_root})
            filenames_filter = get_filenames_for_datasets(dataset_config['sheet_name'], data_root, names)
            dataset_args.update({"filenames_filter": filenames_filter})
        if pass_dictionaries is not None:
            dataset_args.update({"pass_dictionaries": pass_dictionaries})
        dataset_args.update({'data_type': data_type});
        dataset_args.update({'data_cache_tranform': data_cache_tranform});
        dataset = build_dataset_from_input(**dataset_args)

        logger.debug("Transforms %s / %s :\n%s", idx + 1, len(datasets_config), train_transforms)
        datasets.append(dataset)
        logger.info(
            "Data size for %s (%s/%s): %s.",
            dataset_config.text_description,  # type: ignore
            idx + 1,
            len(datasets_config),
            len(dataset),
        )

    return datasets

def build_benchmark_datasets_from_environment(
    env,
    datasets_config: List[DictConfig],
    lists_root: Optional[PathOrString] = None,
    data_sheet: Optional[PathOrString] = None,
    initial_images: Optional[Union[List[pathlib.Path], None]] = None,
    initial_kspaces: Optional[Union[List[pathlib.Path], None]] = None,
    pass_text_description: bool = True,
    pass_dictionaries: Optional[Dict[str, Dict]] = None,
):
    percentages = env.cfg.benchmark.percentages;
    repeats = env.cfg.benchmark.repeats;
    
    dataset_per_percentages = dict();
    for p in percentages:
        dataset_per_repeat = dict();
        for r in range(repeats):
            datasets = []
            for idx, dataset_config in enumerate(datasets_config):
                if pass_text_description:
                    if not "text_description" in dataset_config:
                        dataset_config.text_description = f"ds{idx}" if len(datasets_config) > 1 else None
                else:
                    dataset_config.text_description = None
                if dataset_config.transforms.masking is None:  # type: ignore
                    logger.info(
                        "Masking function set to None for %s.",
                        dataset_config.text_description,  # type: ignore
                    )
                train_transforms, data_cache_tranform = build_transforms_from_environment(env, dataset_config)
                dataset_args = {"transforms": train_transforms, "dataset_config": dataset_config}
                if initial_images is not None:
                    dataset_args.update({"initial_images": initial_images})
                if initial_kspaces is not None:
                    dataset_args.update({"initial_kspaces": initial_kspaces})
                if data_sheet is not None:
                    xls = pd.ExcelFile(data_sheet);
                    sheet_name = dataset_args['dataset_config']['sheet_name']
                    df = pd.read_excel(xls, sheet_name);
                    names = [df.loc[l, 'Name'] for l in np.where(df['New subsets'] == 'val')[0]]
                    data_root = dataset_config['base_path']
                    dataset_args.update({"data_root": data_root})
                    filenames_filter = get_filenames_for_datasets(dataset_config['sheet_name'], data_root, names)
                    #filenames_filter = shuffle(filenames_filter, random_state=42);
                    end_point = int(len(filenames_filter) * p)+1
                    filenames_filter = filenames_filter[:end_point]
                    dataset_args.update({"filenames_filter": filenames_filter})
                if pass_dictionaries is not None:
                    dataset_args.update({"pass_dictionaries": pass_dictionaries})
                dataset_args.update({'data_type': 'bench'});
                dataset_args.update({'data_cache_tranform': data_cache_tranform});
                dataset = build_dataset_from_input(**dataset_args)

                logger.debug("Transforms %s / %s :\n%s", idx + 1, len(datasets_config), train_transforms)
                datasets.append(dataset)
                logger.info(
                    "Data size for %s (%s/%s): %s.",
                    dataset_config.text_description,  # type: ignore
                    idx + 1,
                    len(datasets_config),
                    len(dataset),
                )
            dataset_per_repeat[r] = datasets
        dataset_per_percentages[p] = dataset_per_repeat

    return dataset_per_percentages


def setup_train(
    run_name: str,
    data_sheet_train: Union[pathlib.Path, None],
    data_sheet_validation: Union[pathlib.Path, None],
    base_directory: pathlib.Path,
    cfg_filename: PathOrString,
    force_validation: bool,
    initialization_checkpoint: PathOrString,
    initial_images: Optional[Union[List[pathlib.Path], None]],
    initial_kspace: Optional[Union[List[pathlib.Path], None]],
    noise: Optional[Union[List[pathlib.Path], None]],
    device: str,
    num_workers: int,
    resume: bool,
    machine_rank: int,
    mixed_precision: bool,
    debug: bool,
):
    env = setup_training_environment(
        run_name,
        base_directory,
        cfg_filename,
        device,
        machine_rank,
        mixed_precision,
        debug=debug,
    )

    # Trigger cudnn benchmark and remove the associated cache
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    if initial_kspace is not None and initial_images is not None:
        raise ValueError("Cannot both provide initial kspace or initial images.")
    # Create training data
    training_dataset_args = {"env": env, "datasets_config": env.cfg.training.datasets, "pass_text_description": True}
    pass_dictionaries = {}
    if noise is not None:
        if not env.cfg.physics.use_noise_matrix:
            raise ValueError("cfg.physics.use_noise_matrix is null, yet command line passed noise files.")

        noise = [read_json(fn) for fn in noise]
        pass_dictionaries["loglikelihood_scaling"] = [
            parse_noise_dict(_, percentile=0.999, multiplier=env.cfg.physics.noise_matrix_scaling) for _ in noise
        ]
        training_dataset_args.update({"pass_dictionaries": pass_dictionaries})

    if data_sheet_train is not None:
        training_dataset_args.update({"data_sheet": data_sheet_train})
        # Get the lists_root. Assume now the given path is with respect to the config file.
        lists_root = get_root_of_file(cfg_filename)
        if lists_root is not None:
            training_dataset_args.update({"lists_root": lists_root})
    if initial_images is not None:
        training_dataset_args.update({"initial_images": initial_images[0]})
    if initial_kspace is not None:
        training_dataset_args.update({"initial_kspaces": initial_kspace[0]})

    # Build training datasets
    training_datasets = build_training_datasets_from_environment(**training_dataset_args)
    training_data_sizes = [len(_) for _ in training_datasets]
    logger.info("Training data sizes: %s (sum=%s).", training_data_sizes, sum(training_data_sizes))

    # Create validation data
    if "validation" in env.cfg:
        validation_dataset_args = {
            "env": env,
            "datasets_config": env.cfg.validation.datasets,
            "pass_text_description": True,
        }
        if data_sheet_validation is not None:
            validation_dataset_args.update({"data_sheet": data_sheet_validation})
            lists_root = get_root_of_file(cfg_filename)
            if lists_root is not None:
                validation_dataset_args.update({"lists_root": lists_root})
        if initial_images is not None:
            validation_dataset_args.update({"initial_images": initial_images[1]})
        if initial_kspace is not None:
            validation_dataset_args.update({"initial_kspaces": initial_kspace[1]})
        validation_dataset_args.update({'data_type': 'val'});
        # Build validation datasets
        validation_data = build_training_datasets_from_environment(**validation_dataset_args)
    else:
        logger.info("No validation data.")
        validation_data = None



    # Create the optimizers
    logger.info("Building optimizers.")
    optimizer_params = [{"params": env.engine.model.parameters()}]
    for curr_model_name in env.engine.models:
        # TODO(jt): Can get learning rate from the config per additional model too.
        curr_learning_rate = env.cfg.training.lr
        logger.info("Adding model parameters of %s with learning rate %s.", curr_model_name, curr_learning_rate)
        optimizer_params.append(
            {
                "params": env.engine.models[curr_model_name].parameters(),
                "lr": curr_learning_rate,
            }
        )

    optimizer: torch.optim.Optimizer = str_to_class("torch.optim", env.cfg.training.optimizer)(  # noqa
        optimizer_params,
        lr=env.cfg.training.lr,
        weight_decay=env.cfg.training.weight_decay,
    )  # noqa

    # Build the LR scheduler, we use a fixed LR schedule step size, no adaptive training schedule.
    solver_steps = list(
        range(
            env.cfg.training.lr_step_size,
            env.cfg.training.num_iterations,
            env.cfg.training.lr_step_size,
        )
    )
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        solver_steps,
        env.cfg.training.lr_gamma,
        warmup_factor=1 / 3.0,
        warmup_iterations=env.cfg.training.lr_warmup_iter,
        warmup_method="linear",
    )

    # Just to make sure.
    torch.cuda.empty_cache()

    # Check the initialization checkpoint
    if env.cfg.training.model_checkpoint:
        if initialization_checkpoint:
            logger.warning(
                "`--initialization-checkpoint is set, and config has a set `training.model_checkpoint`: %s. "
                "Will overwrite config variable with the command line: %s.",
                env.cfg.training.model_checkpoint,
                initialization_checkpoint,
            )
            # Now overwrite this in the configuration, so the correct value is dumped.
            env.cfg.training.model_checkpoint = str(initialization_checkpoint)
        else:
            initialization_checkpoint = env.cfg.training.model_checkpoint

    num_workers = env.cfg.training['num_workers'];
    if env.cfg.benchmark.active is True:
        env.engine.benchmark(
            training_datasets,
            env.experiment_dir,
            percentages = env.cfg.benchmark.percentages,
            repeats = env.cfg.benchmark.repeats,
            benchmark_datasets=validation_data,
            start_with_validation=force_validation,
            num_workers=num_workers,
        );
    else:
        env.engine.train(
            optimizer,
            lr_scheduler,
            training_datasets,
            env.experiment_dir,
            validation_datasets=validation_data,
            resume=resume,
            initialization=initialization_checkpoint,
            start_with_validation=force_validation,
            num_workers=num_workers,
        )


def train_from_argparse(args: argparse.Namespace):
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a lot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    # Disable Tensorboard warnings.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if args.initialization_images is not None and args.initialization_kspace is not None:
        sys.exit("--initialization-images and --initialization-kspace are mutually exclusive.")
    check_train_val(args.initialization_images, "initialization-images")
    check_train_val(args.initialization_kspace, "initialization-kspace")
    check_train_val(args.noise, "noise")

    set_all_seeds(args.seed)

    run_name = args.name if args.name is not None else os.path.basename(args.cfg_file)[:-5]

    # TODO(jt): Duplicate params
    launch(
        setup_train,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        run_name,
        args.data_sheet_train,
        args.data_sheet_validation,
        args.experiment_dir,
        args.cfg_file,
        args.force_validation,
        args.initialization_checkpoint,
        args.initialization_images,
        args.initialization_kspace,
        args.noise,
        args.device,
        args.num_workers,
        args.resume,
        args.machine_rank,
        args.mixed_precision,
        args.debug,
    )
