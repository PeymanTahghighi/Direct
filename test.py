# coding=utf-8
# Copyright (c) DIRECT Contributors
"""DIRECT Command-line interface.

This is the file which builds the main parser.
"""

import argparse
from torch.utils.tensorboard import summary
import pathlib
import platform

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

    args = root_parser.parse_args(['train','--cfg', 'configs/base_varnet.yaml'])
   # args = root_parser.parse_args(['predict','--cfg', 'configs/base_varnet_predict.yaml', '--checkpoint', 'experiments/base_varnet/model_500000.pt', '--output_directory', 'test'])
    print(args);
    args.subcommand(args)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import h5py
    # import numpy
    # with h5py.File('inference\\dircn_equispaced_inference_equispaced_train\\skmtea\\E1_MTR_005.h5') as f:
    #     rec = numpy.array(f['reconstruction']);
    #     for i in range(rec.shape[0]):
    #         plt.figure();
    #         plt.imshow(rec[i], cmap= 'gray');
    #     plt.show();
    main()