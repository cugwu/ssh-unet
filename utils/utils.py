# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch
from monai.losses import DiceCELoss


def plot_graph(contents, xlabel, ylabel, savename):
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['red', 'blue']
    for i in range(len(contents)):
        axs.plot(contents[i], linestyle="-", label=ylabel[i], color=colors[i])
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    fig.legend()
    # plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    fig.savefig("%s.png" % (savename))
    # tikz.save("%s.tex" % (savename))
    plt.close()


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
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
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class DeepDiceLoss(torch.nn.Module):
    def __init__(self, dice_loss, deep_supervision, deep_supr_num):
        super().__init__()
        self.dice_loss = dice_loss
        if deep_supervision:
            self.num_heads = deep_supr_num+ 1
        else:
            self.num_heads = 0

    def forward(self, logits, target):
        if self.num_heads == 0:
            loss = self.dice_loss(logits, target)
        else:
            avg_loss = AverageMeter()
            for h in range(self.num_heads):
                avg_loss.update((self.dice_loss(logits[:, h], target)).item())
            loss = avg_loss.avg

        return loss
