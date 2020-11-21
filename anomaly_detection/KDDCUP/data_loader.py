import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
class KDD99Loader(object):
    def __init__(self, data,labels):
        self.data=data
        self.labels=labels

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.data.shape[0]


    def __getitem__(self, index):
        return np.float32(self.data[index]), np.float32(self.labels[index])

        

def get_loader(data_path, batch_size,rng=np.random.RandomState(1)):
    """Build and return data loader."""

    data = np.load(data_path)

    labels = data["kdd"][:, -1]
    features = data["kdd"][:, :-1]
    N, D = features.shape

    # rng = np.random.RandomState(1)
    inds = rng.permutation(N)

    train = features[inds[:N // 2]]
    train_labels = labels[inds[:N // 2]]

    train = train[train_labels == 0]
    train_labels = train_labels[train_labels == 0]

    test = features[inds[N // 2:]]
    test_labels = labels[inds[N // 2:]]

    dataset = KDD99Loader(train,train_labels)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,drop_last=True)
    dev_dataset = KDD99Loader(test,test_labels)
    dev_loader = DataLoader(dataset=dev_dataset,
                             batch_size=batch_size,
                             shuffle=False,drop_last=False)


    return data_loader,dev_loader
