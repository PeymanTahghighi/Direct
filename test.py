# coding=utf-8
# Copyright (c) DIRECT Contributors
"""DIRECT Command-line interface.

This is the file which builds the main parser.
"""

import argparse
from torch.utils.tensorboard import summary
import pickle

def main():

    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    a = pickle.load(open('val_cache_cc_new.ch', 'rb'));
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

    args = root_parser.parse_args(['train','--cfg', 'projects/cvpr2022_recurrentvarnet/fastmri/AXT1_brain/configs/base_varnet.yaml', '--force-validation'])
    print(args);
    args.subcommand(args)


if __name__ == "__main__":
    main()