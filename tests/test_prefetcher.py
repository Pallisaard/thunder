import time
from thunder.dataset import Dataset
from thunder.dataloader import (
    create_dataloader,
    create_jax_dataloader,
    create_numpy_dataloader,
    create_list_dataloader,
    JaxDataLoader,
    NumpyDataLoader,
    ListDataLoader,
)
from thunder.prefetcher import Prefetcher

import numpy as np
import jax.numpy as jnp


class FloatDataset(Dataset):
    data = list(range(10))

    def getitem(self, index: int):
        return self.data[index]

    def len(self) -> int:
        return len(self.data)


def test_prefetcher_factor_1_nworkers_1():
    dl = create_list_dataloader(FloatDataset(), batch_size=1)
    factor = 1
    n_workers = 1

    pf = Prefetcher(dl, prefetch_count=1, num_workers=1)
    for i, item in zip(range(10), pf):
        q_size = min(factor, 10 - i + (n_workers - 1))
        time.sleep(0.05)
        print(i, item)
        assert pf.queue.qsize() == q_size
        assert item == [i]


def test_prefetcher_factor_2_nworkers_1():
    dl = create_list_dataloader(FloatDataset(), batch_size=1)
    factor = 2
    n_workers = 1

    pf = Prefetcher(dl, prefetch_count=2, num_workers=1)
    for i, item in zip(range(10), pf):
        q_size = min(factor, 10 - i + (n_workers - 1))

        time.sleep(0.05)
        print(i, item)
        assert pf.queue.qsize() == q_size
        assert item == [i]


def test_prefetcher_factor_1_nworkers_2():
    dl = create_list_dataloader(FloatDataset(), batch_size=1)
    factor = 1
    n_workers = 2

    pf = Prefetcher(dl, prefetch_count=1, num_workers=2)
    for i, item in zip(range(10), pf):
        q_size = min(factor, 10 - i + (n_workers - 1))
        time.sleep(0.05)
        print(i, item)
        assert pf.queue.qsize() == q_size
        assert item == [i]


def test_prefetcher_factor_2_nworkers_2():
    dl = create_list_dataloader(FloatDataset(), batch_size=1)
    factor = 2
    n_workers = 2

    pf = Prefetcher(dl, prefetch_count=2, num_workers=2)
    for i, item in zip(range(10), pf):
        q_size = min(factor, 10 - i + (n_workers - 1))
        time.sleep(0.05)
        print(i, item)
        assert pf.queue.qsize() == q_size
        assert item == [i]
