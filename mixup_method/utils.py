from __future__ import division

import os
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision
import wandb


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


cifar100 = {
    "mean": np.array([x / 255 for x in [129.3, 124.1, 112.4]]),
    "std": np.array([x / 255 for x in [68.2, 65.4, 70.4]]),
}


imagenet = {
    "mean": np.array([0.485, 0.456, 0.406]),
    "std": np.array([0.229, 0.224, 0.225]),
}


def inverse_normalize(x: torch.Tensor, dataset="cifar100"):
    if dataset == "cifar100":
        return torchvision.transforms.Normalize(
            -cifar100["mean"] / cifar100["std"], 1.0 / cifar100["std"]
        )(x)
    elif dataset == "imagenet":
        return torchvision.transforms.Normalize(
            -imagenet["mean"] / imagenet["std"], 1.0 / imagenet["std"]
        )(x)


def normalize_img(inputs, dataset="cifar100"):
    pass


def wandb_log_csv(filepath):
    """
    Log CSV file to wandb

    :param filepath: string
    :return:
    """
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        wandb.log(dict(row.items()))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def exp_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Exp learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining ** 0.9 * initial_value

    return func


def create_val_folder(data_set_path):
    """
    Used for Tiny-imagenet dataset
    Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
    This method is responsible for separating validation images into separate sub folders,
    so that test and val data can be read by the pytorch dataloaders
    """
    path = os.path.join(
        data_set_path, "val/images"
    )  # path where validation data is present now
    filename = os.path.join(
        data_set_path, "val/val_annotations.txt"
    )  # file where image2class mapping is present
    fp = open(filename, "r")
    data = fp.readlines()

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = os.path.join(path, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


if __name__ == "__main__":
    import argparse
    import urllib
    import zipfile

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    extract_dir = "./"

    zip_path, _ = urllib.request.urlretrieve(url)
    print("Download completed.")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(extract_dir)
    parser = argparse.ArgumentParser(
        description="Rearrage Tiny-ImageNet folder to a readable one for torch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="./tiny-imagenet-200", help="data folder"
    )
    args = parser.parse_args()
    create_val_folder(args.data_dir)
