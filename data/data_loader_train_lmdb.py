from os import path
import lmdb
import msgpack
import numpy as np
import six
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LMDB(Dataset):
    def __init__(self, db_path, transform=None, mask=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.length = msgpack.loads(txn.get(b"__len__"))
            self.keys = msgpack.loads(txn.get(b"__keys__"))
            self.classnum = msgpack.loads(txn.get(b"__classnum__"))

        self.mask = None
        if mask is not None:
            self.mask = np.load(mask)

        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        if self.mask is not None:
            index = self.mask[index]
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def _get_label(self, index):
        if self.mask is not None:
            index = self.mask[index]
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = msgpack.loads(byteflow)
        target = unpacked[1]

        return target

    def __len__(self):
        if self.mask is None:
            return self.length
        else:
            return len(self.mask)

    def get_targets(self):
        targets = []
        for idx in range(self.length):
            target = self._get_label(idx)
            targets.append(target)

        return np.asarray(targets)


class LMDBDataLoader(DataLoader):
    def __init__(self, config, lmdb_path, train=True, mask=None, meta_train=False):
        transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(0.5 if train else 0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self._dataset = LMDB(lmdb_path, transform, mask)

        if meta_train:
            batch_size = len(self._dataset)
        else:
            batch_size = config.batch_size

        super(LMDBDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=train,
            pin_memory=config.pin_memory,
            num_workers=config.workers,
            drop_last=train,
        )

    def class_num(self):
        return self._dataset.classnum
