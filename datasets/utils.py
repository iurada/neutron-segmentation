import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG

class SeededDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=None, 
                 sampler=None, 
                 batch_sampler=None, 
                 num_workers=0, collate_fn=None, 
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, 
                 generator=None, *, prefetch_factor=None, persistent_workers=False, 
                 pin_memory_device=""):
        
        if not CONFIG.use_nondeterministic:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            generator = torch.Generator()
            generator.manual_seed(CONFIG.seed)

            worker_init_fn = seed_worker
        
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, 
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, 
                         pin_memory_device=pin_memory_device)

