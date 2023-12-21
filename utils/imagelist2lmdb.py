#########################################################
# https://github.com/vitoralbiero/face_analysis_pytorch
#########################################################
from os import path
from collections import defaultdict
import lmdb
import msgpack
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class ImageListRaw(ImageFolder):
    def __init__(self, image_list):
        image_names = pd.read_csv(image_list, header=None)
        self.samples = np.asarray(image_names).squeeze()
        self.targets = self.get_labels()
        self.classnum = np.max(self.targets) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with open(self.samples[index], "rb") as f:
            img = f.read()

        return img, self.targets[index]

    def get_labels(self):
        id_dict = defaultdict(int)
        for im in tqdm(self.samples):
            identity = im.split("/")[-2]
            id_dict[identity] += 1
        labels = []
        for i, (k, value) in enumerate(id_dict.items()):
            labels += [i] * value
        return labels


class CustomRawLoader(DataLoader):
    def __init__(self, workers, image_list):
        self._dataset = ImageListRaw(image_list)

        super(CustomRawLoader, self).__init__(
            self._dataset, num_workers=workers, collate_fn=lambda x: x
        )


def list2lmdb(
    image_list,
    dest,
    file_name,
    num_workers=16,
    write_frequency=5000,
):
    print("Loading dataset from %s" % image_list)
    data_loader = CustomRawLoader(num_workers, image_list)

    name = f"{file_name}.lmdb"
    lmdb_path = path.join(dest, name)
    isdir = path.isdir(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    # sigmas = np.linspace(2, 16, 8).astype(int)
    sigmas = [10]

    image_size = 224
    size = len(data_loader.dataset) * image_size * image_size * 3

    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(len(data_loader.dataset))
    txn = db.begin(write=True)
    for idx, data in tqdm(enumerate(data_loader)):
        image, label = data[0]
        txn.put(
            "{}".format(idx).encode("ascii"), msgpack.dumps((image, int(label)))
        )
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    idx += 1

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))
        txn.put(b"__classnum__", msgpack.dumps(int(data_loader.dataset.classnum)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", "-i", help="List of images.")
    parser.add_argument("--workers", "-w", help="Workers number.", default=16, type=int)
    parser.add_argument("--destination", "-d", help="Path to save the lmdb file.")
    parser.add_argument("--file_name", "-n", help="lmdb file name.")
    parser.add_argument("--write_frequency", "-wf", help="write frequency.", default=50000, type=int)
    args = parser.parse_args()

    list2lmdb(
        args.image_list,
        args.dest,
        args.file_name,
        args.workers,
        args.write_frequency
    )
