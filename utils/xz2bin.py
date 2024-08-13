from tqdm import tqdm
import argparse
import lzma
import pickle
from os import path, makedirs


def convert_to_bin(file_path, destination, dataset):
    makedirs(destination, exist_ok=True)
    with lzma.open(file_path, "rb") as fb:
        imgs, issame = pickle.load(fb, encoding="bytes")
    with open(f"{destination}/{dataset}.bin", "wb") as f:
        pickle.dump((imgs, issame), f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    for dataset in tqdm(args.datasets):
        bin_path = f"{args.xz_folder}/{dataset}.xz"
        convert_to_bin(bin_path, args.destination, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get test images from bin files')
    parser.add_argument('--xz_folder', "-f", type=str, help='folder that contains .xz files')
    parser.add_argument('--destination', "-d", type=str, help='destination', default="../test_sets")
    parser.add_argument('--datasets', "-l", type=str, nargs="+",
                        help='dataset to extract',
                        default=["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "hadrian", "eclipse"])
    args = parser.parse_args()
    main(args)
