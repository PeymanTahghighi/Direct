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
from skimage.filters import prewitt, threshold_otsu, sobel, roberts, scharr,threshold_li
from skimage.morphology import binary_closing
import numpy as np
from copy import copy

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

    #args = root_parser.parse_args(['train','--cfg', 'configs/base_varnet.yaml', '--validation-data-type', 'equispaced'])
    #args = root_parser.parse_args(['train','--cfg', 'configs/metamodel.yaml', '--metamodel'])
    args = root_parser.parse_args(['predict','--cfg', 'configs/base_varnet_predict.yaml', '--checkpoint', 'experiments/base_varnet/model_500000.pt', '--output_directory', 'test'])
    print(args);
    args.subcommand(args)

def normalize(data, mi, ma):
       
    data -= mi
    data /= ma

    return data;

if __name__ == "__main__":


    # import matplotlib.pyplot as plt
    # import h5py
    # import numpy
    # ls = SSIMLoss();
    # with h5py.File('inference/dircn_equispaced_inference_equispaced_train/brain/file_brain_AXT2_208_2080561.h5') as f:
    #     rec1 = numpy.array(f['reconstruction']);
    #     tar1 = numpy.array(f['target']);
    
    # with h5py.File('inference/recurrentvarnet_equispaced_inference_equispaced_train/brain/file_brain_AXT2_208_2080561.h5') as f:
    #     rec2 = numpy.array(f['reconstruction']);
    #     tar2 = numpy.array(f['target']);
    
    # mi1 = rec1.min();
    # mi2 = rec2.min();
    # mi = (mi1 + mi2) / 2;

    # ma1 = rec1.max();
    # ma2 = rec2.max();
    # ma = (ma1 + ma2) / 2;

    
    # rec1 = normalize(rec1, mi, ma);
    # rec2 = normalize(rec2, mi, ma);
    # tar1 = normalize(tar1, mi, ma);
    # tar2 = normalize(tar2, mi, ma);

    # mi = rec1.min();
    # ma = rec1.max();

    # mi = rec2.min();
    # ma = rec2.max();

    # mi = tar1.min();
    # ma = tar1.max();

    # mi = tar2.min();
    # ma = tar2.max();



    # ssim_between_targets = fastmri_ssim(torch.from_numpy(tar1).unsqueeze(1), torch.from_numpy(tar2).unsqueeze(1));
    # ssim_between_rec = fastmri_ssim(torch.from_numpy(rec1).unsqueeze(1), torch.from_numpy(rec2).unsqueeze(1));
    # ssim_between_one = fastmri_ssim(torch.from_numpy(rec1).unsqueeze(1), torch.from_numpy(tar1).unsqueeze(1));
    # ssim_between_two = fastmri_ssim(torch.from_numpy(rec2).unsqueeze(1), torch.from_numpy(tar2).unsqueeze(1));

    # import matplotlib.pyplot as plt
    # import h5py
    # import numpy
    # ls = SSIMLoss();
    # with h5py.File('test/file_brain_AXT1_202_2020217.h5') as f:
    #     rec = numpy.array(f['reconstruction']);
    #     tar = numpy.array(f['target']);
    
    # mi_r = rec.min();
    # ma_r = rec.max();

    # mi_t = tar.min();
    # ma_t = tar.max();


    # ssim = fastmri_ssim(torch.from_numpy(rec).unsqueeze(1), torch.from_numpy(tar).unsqueeze(1));
    

    # print(f'ssim {ssim}');
        # for i in range(rec.shape[0]):
        #     fig, ax = plt.subplots(1,3);
        #     cur_rec = copy(rec[i]);

        #     edges_thresh = cur_rec > threshold_li(cur_rec)
        #     footprint= np.ones((1, 51))
        #     edges_close = binary_closing(edges_thresh, footprint);
        #     footprint= np.ones((21, 1))
        #     edges_close = binary_closing(edges_close, footprint);

        #     mask = np.ones_like(cur_rec);

        #     for r in range((mask.shape[0])):
        #         if np.sum(edges_close[r,:]) > 0:
        #             mask_row = np.where(edges_close[r,:] > 0);
        #             first_col = mask_row[0][0];
        #             last_col = mask_row[0][-1];
        #             mask[r,:first_col] = 0;
        #             mask[r,last_col:] = 0;
        #         else:
        #             mask[r,:]=0

        #     rec[i] = rec[i] * mask;
        #     tar[i] = tar[i] * mask;
        #     ax[0].imshow(cur_rec, cmap = 'gray');
        #     ax[0].set_title('rec');
        #     ax[1].imshow(edges_close, cmap = 'gray');
        #     ax[2].imshow(rec[i], cmap = 'gray');
        #     plt.show();

        

        

    main()