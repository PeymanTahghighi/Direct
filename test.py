# coding=utf-8
# Copyright (c) DIRECT Contributors
"""DIRECT Command-line interface.

This is the file which builds the main parser.
"""

import argparse
from torch.utils.tensorboard import summary
import pathlib
import platform
import torch
from direct.functionals.challenges import fastmri_ssim
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """SSIM loss module as implemented in [1]_.

    Parameters
    ----------
    win_size: int
        Window size for SSIM calculation. Default: 7.
    k1: float
        k1 parameter for SSIM calculation. Default: 0.1.
    k2: float
        k2 parameter for SSIM calculation. Default: 0.03.

    References
    ----------

    .. [1] https://github.com/facebookresearch/fastMRI/blob/master/fastmri/losses.py

    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03) -> None:
        """Inits :class:`SSIMLoss`.

        Parameters
        ----------
        win_size: int
            Window size for SSIM calculation. Default: 7.
        k1: float
            k1 parameter for SSIM calculation. Default: 0.1.
        k2: float
            k2 parameter for SSIM calculation. Default: 0.03.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(self, input_data: torch.Tensor, target_data: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SSIMloss`.

        Parameters
        ----------
        input_data : torch.Tensor
            2D Input data.
        target_data : torch.Tensor
            2D Target data.
        data_range : torch.Tensor
            Data range.

        Returns
        -------
        torch.Tensor
        """
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(input_data, self.w)
        uy = F.conv2d(target_data, self.w)
        uxx = F.conv2d(input_data * input_data, self.w)
        uyy = F.conv2d(target_data * target_data, self.w)
        uxy = F.conv2d(input_data * target_data, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return S.mean()

def main():

    plt = platform.system()
    if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
    """Console script for direct."""
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Direct CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular imports
    from direct.cli.predict import register_parser as register_predict_subcommand
    from direct.cli.train import register_parser as register_train_subcommand
    from direct.cli.upload import register_parser as register_upload_subcommand

    # Training images related commands.
    register_train_subcommand(root_subparsers)
    # Inference images related commands.
    register_predict_subcommand(root_subparsers)
    # Data related comments.
    register_upload_subcommand(root_subparsers)

    args = root_parser.parse_args(['predict','--cfg', 'configs/base_varnet.yaml', '--checkpoint', 'model_dircn.pt', '--output_directory', 'test'])
   # args = root_parser.parse_args(['predict','--cfg', 'configs/base_varnet_predict.yaml', '--checkpoint', 'experiments/base_varnet/model_500000.pt', '--output_directory', 'test'])
    print(args);
    args.subcommand(args)

if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    # import h5py
    # import numpy
    # ls = SSIMLoss();
    # with h5py.File('inference\\dircn_equispaced_inference_equispaced_valid\\brain\\file_brain_AXFLAIR_200_6002428.h5') as f:
    #     rec = torch.from_numpy(numpy.array(f['reconstruction']));
    #     tar = torch.from_numpy(numpy.array(f['target']));

    #     # rec = rec;
    #     # rec -= rec.min()
    #     # rec /= rec.max()

    #     # tar = tar;
    #     # tar -= tar.min()
    #     # tar /= tar.max()

    #     ssim = fastmri_ssim(tar.unsqueeze(1), rec.unsqueeze(1));
    #     print(ssim);

    main()