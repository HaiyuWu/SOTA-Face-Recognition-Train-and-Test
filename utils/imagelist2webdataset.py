import multiprocessing
import os
import random
import datetime
import json
import tqdm
from multiprocessing import Process
from torchvision import datasets
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import ImageFolder
from webdataset import TarWriter


queue = multiprocessing.Queue(10240)
dataset_images_num=0
def make_wids_json(pattern,samples_per_shards,shard_ids):
    data = {
            "wids_version": 1,
            # optional immediate shardlist
            "shardlist": [                         
            ],       
    }

    data_items = data["shardlist"]
    for id in shard_ids: 
        data_item={"url": pattern % id,"nsamples": len(samples_per_shards[id])}
        data_items.append(data_item)
    json_file = os.path.join(os.path.dirname(os.path.dirname(pattern)),"datasets_webdataset.json")
    with open(json_file, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
def print_progress():
    write_num = 0
    write_bar = tqdm.tqdm(total=dataset_images_num)
    more = True
    while more:
        deq = queue.get()
        if deq is None:
            more = False
        else:            
            write_num += 1
            if write_num % 1000 == 0:
                write_bar.update(1000)
    print(f"[{datetime.datetime.now()}] print_progress end.")
def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    # random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    #wids json create
    make_wids_json(pattern,samples_per_shards,shard_ids)

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    bar_progress = Process(target=print_progress)        
    for p in processes:
        p.start()
    bar_progress.start()
    # print_progress()
    for p in processes:
        p.join()          
    queue.put(None)
    bar_progress.join()
    
def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    # print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    stream = TarWriter(fname, **kwargs)
    size = 0
    for item in samples:
        size += stream.write(map_func(item))      
        queue.put(1)
    stream.close()
    # print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")
    return size


if __name__ == "__main__":
    root = ""
    dst_tar = "/tar/web4m-%06d.tar"
    items = []

    dataset = open(root,"r").readlines()
    dataset_images_num = len(dataset)
    for i in range(dataset_images_num):
        items.append(dataset[i].replace("\n",""))
    
    def map_func(item):
        name, class_idx = item.split(" ")
        # print(name)
        with open(os.path.join(name), "rb") as stream:
            image = stream.read()
        sample = {
            "__key__": os.path.splitext(os.path.basename(name))[0],
            "jpg": image,
            "cls": int(class_idx)
        }
        return sample

    make_wds_shards(
        pattern=dst_tar,
        num_shards=1024, # 设置分片数量
        num_workers=8, # 设置创建wds数据集的进程数
        samples=items,
        map_func=map_func,
    )