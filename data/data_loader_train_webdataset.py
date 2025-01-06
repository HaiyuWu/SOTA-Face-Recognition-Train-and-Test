from os import path
import wids
import msgpack
import numpy as np
import six
import torch
from PIL import Image
import queue as Queue
import pandas as pd
import threading
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .dist import DistributedSampler, get_dist_info
from .data_augmentor import Augmenter


class WebDataset(Dataset):
    def __init__(self, db_path, transform=None, mask=None, label_map=None, augment=False, fixed_size=None, rank=0):
        self.dataset = wids.ShardListDataset(db_path,cache_size=0)
        # self.label_map = None if label_map is None else np.load(label_map, allow_pickle=True).item()
        self.augmenter = None if not augment else Augmenter(0.3, 0.3, 0.3)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.dataset[index]
        img = sample[".jpg"]
        target = sample[".cls"]
        if self.augmenter is not None:
            img = self.augmenter.augment(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.dataset.total_length

    def get_labels(self):
        id_dict = defaultdict(int)
        for im in self.samples:
            identity = im.split("/")[-2]
            id_dict[identity] += 1
        labels = []
        for i, (k, value) in enumerate(id_dict.items()):
            labels += [i] * value
        return labels


class WebDataLoader(object):
    def __init__(self, config, train=True, seed=2048):
        transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        rank, world_size = get_dist_info()
        # 读取数据集 注释文件
        dataset_info = open(path.join(path.dirname(config.train_source),"dataset_info.txt")).readlines()
        ids = dataset_info[0].split(" ")[0].split(":")[1]    
        self.classnum = int(ids)
        self.img_length = config.num_ims
        self._dataset = WebDataset(config.train_source,transform=transform,augment=config.augment)

        samplers = wids.DistributedChunkedSampler(self._dataset,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=False,
                                        seed=seed)
                                        # ,
                                        # fixed_size=config.fixed_size)

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
        return self.classnum

    def get_loader(self):
        return self.loader

    def get_length(self):
        return self.img_length


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
if __name__ == '__main__':
    # data_json = "/datasets/faceid/web4m/datasets_webdataset.json"
    # dataset_info = open(path.join(data_json,dataset_info.txt)).readlines()
    # ids = dataset_info[0].split(" ")[0].split(":")[1]    
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((112, 112)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     ]
    # )
    # dataset = WebDataset(data_json,transform)
    # load = DataLoader(dataset,batch_size=1)
    # for img,label in enumerate(load):
    #     print(label)
    pass
