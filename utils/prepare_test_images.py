from tqdm import tqdm
import argparse
import lzma
import numpy as np
import pickle
from os import path, makedirs
import cv2


def load_xz(file_path, image_size=(112, 112)):
    with lzma.open(file_path, "rb") as fb:
        imgs, issame = pickle.load(fb, encoding="bytes")

    dataset = []
    for i in range(len(imgs)):
        # decode from byte
        img = cv2.imdecode(np.asarray(bytearray(imgs[i])), cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_size)
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset.append(img)
    dataset = np.asarray(dataset).transpose((0, 3, 1, 2))
    return dataset, issame


def convert_to_test(images, issame, dataset, destination):
    save_folder = f"{destination}/{dataset}"
    if not path.exists(save_folder):
        makedirs(save_folder)
    np.save(f"{save_folder}/{dataset}.npy", images)
    np.savetxt(f"{save_folder}/issame.txt", np.array(issame).astype(int), fmt="%s")


def main(args):
    for dataset in tqdm(args.datasets):
        bin_path = f"{args.xz_folder}/{dataset}.xz"
        images, issame = load_xz(bin_path, image_size=(112, 112))
        convert_to_test(images, issame, dataset, args.destination)
        # save all the images
        # images = images.transpose(0, 2, 3, 1)
        # logger = [0, 0]
        # for i in tqdm(range(len(issame))):
        #     if issame[i]:
        #         dir_num = logger[0]
        #         logger[0] += 1
        #         save_folder = f"{args.destination}/{dataset}/gen/{str(dir_num).zfill(4)}"
        #     else:
        #         dir_num = logger[1]
        #         save_folder = f"{args.destination}/{dataset}/imp/{str(dir_num).zfill(4)}"
        #         logger[1] += 1
        #     if not path.exists(save_folder):
        #         makedirs(save_folder)
        #     io.imsave(f"{save_folder}/{dir_num}_0.jpg", images[2 * i])
        #     io.imsave(f"{save_folder}/{dir_num}_1.jpg", images[2 * i + 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get test images from bin files')
    parser.add_argument('--xz_folder', "-f", type=str, help='folder that contains .xz files')
    parser.add_argument('--destination', "-d", type=str, help='destination', default="../test_sets")
    parser.add_argument('--datasets', "-l", type=str, nargs="+",
                        help='dataset to extract',
                        default=["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "hadrian", "eclipse"])
    args = parser.parse_args()
    main(args)
