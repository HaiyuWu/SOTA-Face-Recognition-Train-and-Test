import argparse
import os
from tqdm import tqdm
import os.path as path
from glob import glob


def main(source,
         destination,
         dest_name,
         file_type):
    folder_paths = sub_folders(source)
    parent_folder = os.path.dirname(source)
    if dest_name is None:
        dest_name = path.split(source)[1]
    if dest_name == "":
        dest_name = path.split(source)[0].split("/")[-1]
        parent_folder = os.path.dirname(path.split(source)[0])
    if not destination:
        file_path = path.join(parent_folder, f"{dest_name}.txt")
    else:
        file_path = path.join(destination, f"{dest_name}.txt")
    if path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "a+") as f:
        for folder_path in tqdm(folder_paths):
            for im_path in glob(folder_path + f"/*{file_type}"):
                f.write(path.abspath(im_path) + "\n")
    print(f"Image paths has been saved to {file_path}")


def sub_folders(source):
    folders = []
    find_folder_path(source, folders)
    return folders


def find_folder_path(source, folder_paths):
    if not os.path.isdir(source):
        return True
    for sub_dir in os.listdir(source):
        next_path = path.join(source, sub_dir)
        flag = find_folder_path(next_path, folder_paths)
        if flag:
            folder_paths.append(source)
            return False
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Store the image paths")
    parser.add_argument("--source", "-s", help="folder of the images.")
    parser.add_argument("--destination", "-d", help="destination of file.", type=str, default=None)
    parser.add_argument("--saved_file_name", "-sfn", help="name of file to save.", type=str, default=None)
    parser.add_argument("--file_type", "-end_with", help="the type of the aim file.", type=str, default="jpg")

    args = parser.parse_args()

    main(args.source,
         args.destination,
         args.saved_file_name,
         args.file_type)
