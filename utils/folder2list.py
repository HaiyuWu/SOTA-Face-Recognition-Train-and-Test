import argparse
from os import listdir, path

import numpy as np


def convert(main_folder, output):
    ret = []

    for  class_folder in listdir(main_folder):
        class_folder_path = path.join(main_folder, class_folder)

        for img_name in listdir(class_folder_path):
            image_path = path.join(class_folder, img_name)
            ret.append([image_path])

    np.savetxt(output, ret, delimiter=" ", fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Folder with classes subfolders to a file to train."
    )
    parser.add_argument("--folder", "-f", help="Folder to convert.")
    parser.add_argument("--output", "-o", help="Output file.")

    args = parser.parse_args()

    convert(args.folder, args.output)