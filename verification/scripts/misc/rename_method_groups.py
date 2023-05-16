"""
Small script to rename the method group in prediction HDF5 files.
Bent Harnist (FMI) -  2022-2023
"""
import argparse

import h5py
from tqdm import tqdm

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "hdf5-path", type=str, help="Path to the HDF5 archive to modify."
    )
    argparser.add_argument(
        "-o", "--old-name", type=str, help="Old method group name (to remove)"
    )
    argparser.add_argument(
        "-n", "--new-name", type=str, help="New method group name (to replace with)"
    )
    argparser.add_argument(
        "-pt",
        "--path-template",
        type=str,
        default="{time}/{method}",
        help="Prediction path template inside the HDF5 archive, with {time} and {method} keys.",
    )
    args = argparser.parse_args()

    with h5py.File(args.hdf5_path, "a") as f:
        for t in f.keys():
            old_path = args.path_template.format(time=t, method=args.old_name)
            new_path = args.path_template.format(time=t, method=args.new_name)
            f.move(source=old_path, dest=new_path)

    print(
        f"\n Success!\n Method group in {args.hdf5_path} renamed from {args.old_name} to {args.new_name}."
    )
