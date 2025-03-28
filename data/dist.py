import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler as _DistributedSampler


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def sync_random_seed(seed=None, device="cuda"):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)

    return random_num.item()


class DistributedSampler(_DistributedSampler):
    def __init__(
            self,
            dataset,
            num_replicas=None,  # world_size
            rank=None,  # local_rank
            shuffle=True,
            seed=0,
            fixed_size=None  # New parameter
    ):
        # Initialize base sampler first
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # Initialize seed for consistent shuffling
        self.seed = sync_random_seed(seed)

        # Handle fixed-size sampling
        if fixed_size is not None:
            # Calculate samples per replica and total size
            samples_per_replica = math.ceil(fixed_size / self.num_replicas)
            self.total_size_original = self.total_size  # Save original total_size
            self.total_size = samples_per_replica * self.num_replicas
            self.num_samples = samples_per_replica
            self.fixed_size = fixed_size
        else:
            self.fixed_size = None
            self.total_size_original = self.total_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # If using fixed size sampling
        if self.fixed_size is not None:
            # First extend indices if needed to reach at least fixed_size
            num_repeats = math.ceil(self.total_size / len(indices))
            indices = indices * num_repeats

            # Then trim to exact total_size
            indices = indices[:self.total_size]
        else:
            # Original behavior for full dataset
            num_repeats = math.ceil(self.total_size_original / len(indices))
            indices = (indices * num_repeats)[:self.total_size_original]

        # Verify we have the correct number of indices
        assert len(indices) == (self.total_size if self.fixed_size else self.total_size_original)

        # subsample for this rank
        indices = indices[self.rank:len(indices):self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
