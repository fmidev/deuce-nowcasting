import h5py
import hdf5plugin
from tqdm import tqdm
import argparse


def recurse_tree(file_in, file_out, chunk_size: tuple, compression: str):
    for name, item in tqdm(file_in.items()):
        print(name)
        if isinstance(item, h5py.Group):
            print(name)
            file_out.require_group(name)
            recurse_tree(
                file_in=file_in[name],
                file_out=file_out[name],
                chunk_size=chunk_size,
                compression=compression,
            )
        elif isinstance(item, h5py.Dataset):
            data_in = item[:]
            data_in[data_in < 76] = 0

            if compression == "blosc":
                file_out.create_dataset(
                    name,
                    data=data_in,
                    dtype="uint8",
                    chunks=tuple(chunk_size),
                    **hdf5plugin.Blosc(clevel=9, cname="zstd")
                )
            elif compression == "szip":
                file_out.create_dataset(
                    name,
                    data=data_in,
                    dtype="uint8",
                    chunks=tuple(chunk_size),
                    compression="szip",
                    compression_opts=("nn", 8),
                )
            else:
                file_out.create_dataset(
                    name,
                    data=data_in,
                    dtype="uint8",
                    chunks=tuple(chunk_size),
                    compression="gzip",
                    compression_opts=9,
                )
        else:
            return
    return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("--input", type=str, help="input HDF5 file")
    argparser.add_argument("--output", type=str, help="output HDF5 file")
    argparser.add_argument("--chunk_size", nargs="+", type=int)
    argparser.add_argument("--compression", type=str)
    args = argparser.parse_args()
    with h5py.File(args.input, "r") as f:
        with h5py.File(args.output, "a") as g:
            recurse_tree(
                file_in=f,
                file_out=g,
                chunk_size=args.chunk_size,
                compression=args.compression,
            )
