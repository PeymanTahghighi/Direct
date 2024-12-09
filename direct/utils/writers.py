# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import pathlib
from typing import Callable, DefaultDict, Dict, Optional, Union

import h5py  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


def write_output_to_h5(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
    metamodel: bool = False
) -> None:
    """Write dictionary with keys filenames and values torch tensors to h5 files.

    Parameters
    ----------
    output: dict
        Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
        where num_channels is typically 1 for MRI.
    output_directory: pathlib.Path
    volume_processing_func: callable
        Function which postprocesses the volume array before saving.
    output_key: str
        Name of key to save the output to.
    create_dirs_if_needed: bool
        If true, the output directory and all its parents will be created.

    Notes
    -----
    Currently only num_channels = 1 is supported. If you run this function with more channels the first one
    will be used.
    """
    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)
    total_ssim = [];
    if metamodel is False:
        for idx, (prediction, target, scaling_factor, ssim,_, filename) in enumerate(output):
            # The output has shape (slice, 1, height, width)
            if isinstance(filename, pathlib.PosixPath):
                filename = filename.name

            logger.info(f"({idx + 1}/{len(output)}): Writing {output_directory / filename}...")

            reconstruction = prediction.numpy()[:, 0, ...].astype(np.float32)
            target = target.numpy()[:, 0, ...].astype(np.float32)

            if volume_processing_func:
                reconstruction = volume_processing_func(reconstruction)
            with h5py.File(output_directory / filename, "w") as f:
                f.create_dataset(output_key[0], data=reconstruction)
                f.create_dataset(output_key[1], data=target)
                f.create_dataset('scaling_factor', data=scaling_factor)
    else:
        for idx, (prediction, target, ssim, _, filename) in enumerate(output):
            # The output has shape (slice, 1, height, width)
            if isinstance(filename, pathlib.PosixPath):
                filename = filename.name

            logger.info(f"({idx + 1}/{len(output)}): Writing {output_directory / filename}...")

            reconstruction = prediction.numpy()[:, 0, ...].astype(np.float32)
            target = target.numpy()[:, 0, ...].astype(np.float32)

            if volume_processing_func:
                reconstruction = volume_processing_func(reconstruction)
            with h5py.File(output_directory / filename, "w") as f:
                f.create_dataset(output_key[0], data=reconstruction)
                f.create_dataset(output_key[1], data=target)
