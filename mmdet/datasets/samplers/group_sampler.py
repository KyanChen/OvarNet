# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        if hasattr(dataset, 'flag_dataset'):
            self.flag_dataset = dataset.flag_dataset.astype(np.int64)
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            if hasattr(self, 'flag_dataset'):
                # rank, world_size = get_dist_info()
                indice = np.array(indice, dtype=np.int64)
                flag_dataset_tmp = self.flag_dataset[indice]
                num_flag_dataset = np.bincount(flag_dataset_tmp)
                # 把每个卡上每个flag的样本数求出来
                samples_per_flag = {
                    idx_flag: round(self.samples_per_gpu * (v / num_flag_dataset.sum()))
                    for idx_flag, v in enumerate(num_flag_dataset)
                }
                # 补齐样本数
                samples_per_flag[list(samples_per_flag.keys())[0]] = \
                    self.samples_per_gpu - sum(list(samples_per_flag.values())[1:])

                max_group_samps = max(
                    [math.ceil(num_flag_dataset[idx_flag] / samples_per_flag[idx_flag])
                     for idx_flag in range(len(num_flag_dataset))]
                )
                data_flag_indices = {idx_flag: [] for idx_flag in range(len(num_flag_dataset))}
                for flag_i, flag_size in enumerate(num_flag_dataset):
                    data_flag_indice = np.where(flag_dataset_tmp == flag_i)[0]
                    assert len(data_flag_indice) == flag_size
                    data_flag_indice = data_flag_indice.tolist()
                    # 让 index 越界
                    data_flag_indice += data_flag_indice
                    data_flag_indices[flag_i] = data_flag_indice
                indice_rearrange = []
                for i_group in range(max_group_samps):
                    indice_per_gpu = []
                    for i_flag in range(len(num_flag_dataset)):
                        num_samp_of_flag = samples_per_flag[i_flag]
                        indice_per_gpu += data_flag_indices[i_flag][i_group * num_samp_of_flag: (i_group + 1) * num_samp_of_flag]
                    indice_rearrange += indice_per_gpu
                indice = indice[indice_rearrange]
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        # assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        if hasattr(dataset, 'flag_dataset'):
            self.flag_dataset = dataset.flag_dataset.astype(np.int64)

        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                if hasattr(self, 'flag_dataset'):
                    # rank, world_size = get_dist_info()
                    indice = np.array(indice, dtype=np.int64)
                    flag_dataset_tmp = self.flag_dataset[indice]
                    num_flag_dataset = np.bincount(flag_dataset_tmp)
                    # 把每个卡上每个flag的样本数求出来
                    samples_per_flag = {
                        idx_flag: round(self.samples_per_gpu * (v / num_flag_dataset.sum()))
                        for idx_flag, v in enumerate(num_flag_dataset)
                    }
                    # 补齐样本数
                    samples_per_flag[list(samples_per_flag.keys())[0]] = \
                        self.samples_per_gpu - sum(list(samples_per_flag.values())[1:])

                    max_group_samps = max(
                        [math.ceil(num_flag_dataset[idx_flag] / samples_per_flag[idx_flag])
                         for idx_flag in range(len(num_flag_dataset))]
                    )
                    data_flag_indices = {idx_flag: [] for idx_flag in range(len(num_flag_dataset))}
                    for flag_i, flag_size in enumerate(num_flag_dataset):
                        data_flag_indice = np.where(flag_dataset_tmp == flag_i)[0]
                        assert len(data_flag_indice) == flag_size
                        data_flag_indice = data_flag_indice.tolist()
                        # 让 index 越界
                        data_flag_indice += data_flag_indice
                        data_flag_indices[flag_i] = data_flag_indice
                    indice_rearrange = []
                    for i_group in range(max_group_samps):
                        indice_per_gpu = []
                        for i_flag in range(len(num_flag_dataset)):
                            num_samp_of_flag = samples_per_flag[i_flag]
                            indice_per_gpu += data_flag_indices[i_flag][
                                              i_group * num_samp_of_flag: (i_group + 1) * num_samp_of_flag]
                        indice_rearrange += indice_per_gpu
                    indice = indice[indice_rearrange]
                indices.extend(indice)
        # assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        # assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
