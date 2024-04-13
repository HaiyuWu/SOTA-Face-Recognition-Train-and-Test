from os import path
import lmdb
import msgpack
import numpy as np
import six
import torch
from PIL import Image
import queue as Queue
import threading
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .dist import DistributedSampler, get_dist_info
from .data_augmentor import Augmenter


class LMDB(Dataset):
    def __init__(self, db_path, transform=None, mask=None, label_map=None, augment=False):
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

        self.label_map = None if label_map is None else np.load(label_map, allow_pickle=True).item()
        self.augmenter = None if not augment else Augmenter(0.3, 0.3, 0.3)
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
        if self.label_map is not None:
            try:
                target = self.label_map[str(target)]
            except KeyError:
                pass
        if self.augmenter is not None:
            img = self.augmenter.augment(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.mask is None:
            return self.length
        else:
            return len(self.mask)


class LMDBDataLoader(object):
    def __init__(self, config, train=True, seed=2048):
        transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self._dataset = LMDB(config.train_source, transform, config.mask, config.label_map, config.augment)
        rank, world_size = get_dist_info()
        samplers = DistributedSampler(self._dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

        # use DataLoaderX for faster loading
        self.loader = DataLoaderX(
            local_rank=rank,
            dataset=self._dataset,
            batch_size=config.batch_size,
            sampler=samplers,
            num_workers=config.workers,
            pin_memory=config.pin_memory,
            drop_last=train,
        )

    def class_num(self):
        return self._dataset.classnum

    def get_loader(self):
        return self.loader


#################################################################################################
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py#L27
#################################################################################################
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
