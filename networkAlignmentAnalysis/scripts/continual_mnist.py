# Some Code for continual learning with permuted MNIST
import time
from functools import partial
import numpy as np
import scipy as sp
import sklearn
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2 as transforms

from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt

import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(mainPath)

from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis.datasets import get_dataset
from networkAlignmentAnalysis.experiments.registry import get_experiment
from networkAlignmentAnalysis import utils
from networkAlignmentAnalysis import files
from networkAlignmentAnalysis import train


def permute(batch, batch_dim=True, shuffle_idx=None):
    if shuffle_idx is not None:
        original_size = batch[0].shape
        if batch_dim:
            batch[0] = batch[0][:, shuffle_idx]
        else:
            batch[0] = batch[0][shuffle_idx]
        batch[0].reshape(original_size)
    return batch


def add_permutation(dataset, num_pixels=784):
    perm = partial(permute, batch_dim=True, shuffle_idx=torch.randperm(num_pixels))
    if dataset.extra_transform is None:
        dataset.extra_transform = []
    dataset.extra_transform.append(perm)
    return dataset


def update_permutation(dataset, num_pixels=784):
    perm = partial(permute, batch_dim=True, shuffle_idx=torch.randperm(num_pixels))
    dataset.extra_transform[-1] = perm
    return dataset


def do_permuted_round(nets, optimizers, dataset, train_epochs=1, verbose=False):
    parameters = dict(
        verbose=verbose,
        num_epochs=train_epochs,
        alignment=False,
    )
    dataset = update_permutation(dataset)

    # train and test
    train_results = train.train(nets, optimizers, dataset, **parameters)
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device: ", DEVICE)

    model_name = "MLP"
    dataset_name = "MNIST"

    hidden_widths = [200, 200, 200]

    lrs = [1e-1, 3e-2, 1e-2, 3e-3]
    num_replicates = 3

    nets = []
    optimizers = []
    net_lr = []
    for lr in lrs:
        for _ in range(num_replicates):
            net = get_model(model_name, build=True, dataset=dataset_name, hidden_widths=hidden_widths, dropout=0.0, ignore_flag=False)
            net.to(DEVICE)

            optimizer = torch.optim.SGD(net.parameters(), lr=lr)

            nets.append(net)
            optimizers.append(optimizer)
            net_lr.append(lr)

    loader_parameters = dict(
        shuffle=True,
        batch_size=5,
    )
    dataset = get_dataset(dataset_name, build=True, transform_parameters=net, loader_parameters=loader_parameters, device=DEVICE)
    dataset = add_permutation(dataset)

    num_rounds = 200

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    for round in tqdm(range(num_rounds)):
        c_train_res, c_test_res = do_permuted_round(nets, optimizers, dataset, verbose=False)
        train_loss.append(c_train_res["loss"])
        train_accuracy.append(c_train_res["accuracy"])
        test_loss.append(c_test_res["loss"])
        test_accuracy.append(c_test_res["accuracy"])

    loss = torch.stack([torch.tensor(l) for l in test_loss])
    accuracy = torch.stack([torch.tensor(a) for a in test_accuracy])

    type_loss = utils.compute_stats_by_type(loss, len(lrs), 1)[0]
    type_accuracy = utils.compute_stats_by_type(accuracy, len(lrs), 1)[0]

    # print(loss.shape, accuracy.shape, type_loss.shape, type_accuracy.shape)

    cols = mpl.colormaps["Set1"].resampled(len(lrs))
    names = [f"lr={lr}" for lr in lrs]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
    for ii in range(len(lrs)):
        ax[0].plot(range(num_rounds), type_loss[:, ii], c=cols(ii), label=names[ii])
        ax[1].plot(range(num_rounds), type_accuracy[:, ii], c=cols(ii), label=names[ii])

    ax[0].set_xlabel("Rounds of Permuted MNIST")
    ax[1].set_xlabel("Rounds of Permuted MNIST")
    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("accuracy")
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    plt.show()

    print("Hello, add a debugger here to evaluate while testing")


if __name__ == "__main__":
    main()
