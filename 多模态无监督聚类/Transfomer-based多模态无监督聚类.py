# -*- coding: utf-8 -*-
"""
Standalone integration generated from the uploaded original project.

Target dataset layout (per the provided project tree):
    dataset/
      Augsburg/augsburg_gt.mat, augsburg_hsi.mat, augsburg_sar.mat, ...
      Houston/gt.mat, HSI.mat, LiDAR.mat, ...
      MUUFL/gt.mat, HSI.mat, LiDAR.mat, ...
      Trento/gt.mat, HSI.mat, LiDAR.mat, ...

The model/loss/training hyperparameter defaults below are kept from the
project's original config. The only intentional filesystem change is that
--dataset_root points to the unified "dataset" directory above.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Original project rolling-window implementation, kept behaviorally equivalent."""
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")
    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        _asteps[-len(asteps):] = asteps
        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")
        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1
    wsteps = _wsteps
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")
    new_shape = orig_shape
    _window = window.copy()
    _window[_window == 0] = 1
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    new_shape[new_shape < 1] = 1
    shape = new_shape
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

rw = rolling_window


class Processor:
    def prepare_data(self, img_path, gt_path=None):
        img_mat = sio.loadmat(str(img_path))
        img_keys = [k for k in img_mat.keys() if not k.startswith('__')]
        if not img_keys:
            raise KeyError(f"No data variable found in {img_path}")
        img = np.asarray(img_mat[img_keys[0]]).astype('float32')
        if gt_path is None:
            return img
        gt_mat = sio.loadmat(str(gt_path))
        gt_keys = [k for k in gt_mat.keys() if not k.startswith('__')]
        if not gt_keys:
            raise KeyError(f"No GT variable found in {gt_path}")
        gt = np.asarray(gt_mat[gt_keys[0]])
        gt = np.squeeze(gt).astype('int64')
        if gt.ndim != 2:
            raise ValueError(f"Ground truth must be 2-D after squeeze, got {gt.shape} from {gt_path}")
        if img.ndim >= 2 and tuple(img.shape[:2]) != tuple(gt.shape):
            raise ValueError(
                f"Spatial size mismatch: image {img.shape[:2]} vs GT {gt.shape}. "
                f"Check MAT orientation/variable selection: {img_path}"
            )
        return img, gt

    def get_HSI_patches_rw(self, x, gt, ksize, stride=(1, 1), padding='reflect', is_indix=False, is_labeled=True):
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
        n_row_init, n_col_init, _ = x.shape
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x = np.reshape(x, (n_row, n_clm, n_band))
        y = np.reshape(gt, (n_row, n_clm))
        ksizes_ = (ksize[0], ksize[1])
        x_patches = rw(x, ksizes_, axes=(1, 0))
        y_patches = rw(y, ksizes_, axes=(1, 0))
        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
        nonzero_index = None
        if not is_labeled:
            x_patches = x_patches.reshape((n_row_init * n_col_init, n_band, ksize[0], ksize[1]))
            y_patches = y_patches[:, :, i_1, i_2].reshape(-1)
        else:
            nonzero_index = y_patches[:, :, i_1, i_2].nonzero()
            x_patches = x_patches[nonzero_index]
            y_patches = (y_patches[:, :, i_1, i_2])[nonzero_index]
        x_patches = np.transpose(x_patches, [0, 2, 3, 1])
        if is_indix:
            return x_patches, y_patches, nonzero_index
        return x_patches, y_patches

    def standardize_label(self, y):
        classes = np.unique(y)
        standardize_y = copy.deepcopy(y)
        for i in range(classes.shape[0]):
            standardize_y[np.nonzero(y == classes[i])] = i
        return standardize_y


class MultiModalDataset(Dataset):

    def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True):
        self.transform = transform
        p = Processor()
        n_modality = len(src_path)
        modality_list = []
        in_channels = []
        for i in range(n_modality):
            img, gt = p.prepare_data(src_path[i], gt_path)
            x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
            n_samples, n_row, n_col, n_channel = x_patches.shape
            scaler = StandardScaler()
            batch_size = 5000
            # # using incremental / batch for very large data
            for start_id in range(0, x_patches.shape[0], batch_size):
                n_batch = x_patches[start_id: start_id+batch_size].shape[0]
                scaler.partial_fit(x_patches[start_id: start_id+batch_size].reshape(n_batch, -1))
            for start_id in range(0, x_patches.shape[0], batch_size):
                shape = x_patches[start_id: start_id+batch_size].shape
                x_temp = x_patches[start_id: start_id+batch_size].reshape(shape[0], -1)
                x_patches[start_id: start_id+batch_size] = scaler.transform(x_temp).reshape(shape)
            x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
            x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
            modality_list.append(x_tensor)
            in_channels.append(n_channel)
        y = p.standardize_label(y_)
        self.gt_shape = gt.shape
        self.data_size = len(y)
        if is_labeled:
            self.n_classes = np.unique(y).shape[0]
        else:
            self.n_classes = np.unique(y).shape[0] - 1  # remove background
        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)
        self.modality_list = tuple(modality_list)
        self.n_modality = n_modality
        self.in_channels = tuple(in_channels)

    def __getitem__(self, idx):
        x_list = []
        for i in range(self.n_modality):
            x = self.modality_list[i][idx]
            if self.transform is not None:
                x_1, x_2 = self.transform(x)  # # conduct transformation on a single modality
                x_list.append(x_1)
                x_list.append(x_2)
            else:
                x_list.append(x)
        if self.n_modality >= 2 and len(x_list) > 2:  # # when modality >= 2, i.e., 4 augs
            x_list = (x_list[0::2], x_list[1::2])
        if self.n_modality == 1 and len(x_list) == 2:
            x_list = ([x_list[0]], [x_list[1]])
        y = self.y_tensor[idx]
        return x_list, y

    def __len__(self):
        return self.data_size

class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, img):
        prob = np.random.random_sample()
        if prob < 0.5:
            img = np.array(img)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)
            img = torch.from_numpy(img).float()
        return img

class Transforms:
    def __init__(self, size, mean=None, std=None, blur=False):
        self.train_transform = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomChoice([GaussianBlur(3),
                                                 MaskPixels(p=0.2),
                                                 MaskBands(p=0.2)
                                                ], p=[0.4, 0.5, 0.1]),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=3))
        # self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            # torchvision.transforms.Resize(size=(size, size)),
            # MaskBands(),
            # RandomProjectionBands(n_band=200)
            # torchvision.transforms.ToTensor(),
            # MaskBands(p=0.2),
            # RandomProjectionBands(n_band=32),
            # PermuteBands(10)
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class GroupPermuteBands(object):
    """
    shuffle bands into n_groups
    """
    def __init__(self, n_group=3):
        self.n_group = n_group

    def __call__(self, img):
        n_channel = img.size(0)
        n_group_band = int(np.ceil(n_channel / self.n_group))
        for i in range(self.n_group):
            start = i * n_group_band
            end = start + n_group_band
            if end >= n_channel:
                indx = np.arange(start, n_channel)
                indx_ = np.arange(start, n_channel)
            else:
                indx = np.arange(start, end)
                indx_ = np.arange(start, end)
            np.random.shuffle(indx)
            img[indx_] = img[indx]
        # indx_selected = indx[:n_shuffle]
        # select_mask = np.zeros((n_channel, 1, 1))
        # select_mask[indx_selected] = 1
        # img_shuffled = img[indx]

        return img

class MaskPixels(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        n_band, h, w = img.shape
        mask = np.random.binomial(1, self.p, size=(h, w))
        mask = torch.from_numpy(mask).float()
        mask = mask.expand((n_band, h, w))
        img = mask * img
        return img

class MaskBands(object):

    def __init__(self, p=0.5):
        """

        :param p: a band will be masked with probability of p
        """
        self.p = 1. - p

    def __call__(self, img):
        # indx = np.arange(img.shape[0])
        # indx_selected = np.random.choice(indx, self.n_band, replace=False)
        # img = img[indx_selected]

        prob = np.random.binomial(1, self.p, img.shape[0])
        prob = np.reshape(prob, (img.shape[0], 1, 1))
        prob = torch.from_numpy(prob).float()
        # img = img[np.where(prob == 1)]
        img = img * prob
        return img

class RandomProjectionBands(object):

    def __init__(self, n_band=None):
        """
        :param n_band: project to n_band
        """
        self.n_band = n_band

    def __call__(self, img):
        # # n_band * w * h
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        n_band, h, w = img.shape
        if self.n_band is None:
            # self.n_band = np.random.randint(3, n_band//2)
            transformer = random_projection.SparseRandomProjection(n_components='auto')
        else:
            transformer = random_projection.SparseRandomProjection(n_components=self.n_band)
        img_ = img.transpose((1, 2, 0))
        x_2d = img_.reshape((-1, n_band))
        x_2d_ = transformer.fit_transform(x_2d)
        img_new = x_2d_.reshape((h, w, -1)).transpose(2, 0, 1)
        img_new = torch.from_numpy(img_new).float()
        return img_new

class ShufflePixel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        n_band, h, w = img.shape
        img_ = img.view(n_band, -1)
        img_ = img_[torch.randperm(n_band)]
        img = img_.view(n_band, h, w)
        return img

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x  # self.mlp_head(x)

class PatchEmbedding(nn.Module):
    """
    transform different modalities into the same dim
    """

    def __init__(self, n_modalities, in_channels, out_channel):
        """
        :param n_modalities: number of modalities
        :param in_channels: tuple of input channels of multiple modalities, or a single modality
        :param out_channel:
        """
        super(PatchEmbedding, self).__init__()
        self.n_modalities = n_modalities
        self.out_channel = out_channel
        self.in_channels = in_channels
        if not isinstance(self.in_channels, tuple):
            self.in_channels = (self.in_channels,)
        self.layers = nn.ModuleList([nn.Conv2d(self.in_channels[i], out_channel, (3, 3)) for i in range(n_modalities)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_channel) for i in range(n_modalities)])

    def forward(self, x):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        x = [bn(layer(x_i)) for x_i, layer, bn in zip(x, self.layers, self.bn)]
        x = torch.cat(x, dim=-1)
        return x

class ContrastiveHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ContrastiveHead, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp_head(x)
        return x

class ClusteringHead(nn.Module):
    def __init__(self, n_dim, n_class, alpha=1.):
        super(ClusteringHead, self).__init__()
        # Clustering head
        self.alpha = alpha
        # initial_cluster_centers = torch.tensor(torch.randn((n_class, n_dim), dtype=torch.float, requires_grad=True))
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        # torch.nn.init.orthogonal_(self.cluster_centers.data, gain=1)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        """
        :param x: n_batch * n-dim
        :return:
        """
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class Net(nn.Module):
    def __init__(self, n_modalities, in_channels, in_patch_size, common_channel, n_class, dim_emebeding):
        super(Net, self).__init__()
        self.embedding_layer = PatchEmbedding(n_modalities, in_channels, common_channel)
        self.vit = ViT(image_size=(in_patch_size[0]-2, (in_patch_size[1]-2) * 2),  # use 3*3 kernel in embedding layer #(5, 10),
                       # image_size=(in_patch_size[0], in_patch_size[1] * 2),
                       patch_size=1,
                       # num_classes=n_class,
                       dim=512,
                       depth=4,
                       heads=8,
                       mlp_dim=1024,
                       pool='mean',
                       channels=common_channel,
                       dim_head=64,
                       dropout=0.1,
                       emb_dropout=0.1
                       )
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1) ## ContrastiveHead(512, 128)

    def forward(self, x_1, x_2):
        """
        :param x_1, x_2: tuple of modalities, e.g., [aug_1, aug_2]-->
        ([img_rgb, img_hsi, img_sar], [img_rgb, img_hsi, img_sar])
        :return:
        """
        x_1 = self.vit(self.embedding_layer(x_1))  # # concatenated modalities: [batch, n_channel, width, 2*height]
        x_2 = self.vit(self.embedding_layer(x_2))

        y_1 = self.clustering_head(x_1)
        y_2 = self.clustering_head(x_2)

        return y_1, y_2

    def forward_embedding(self, x):
        # h = self.clustering_head(self.vit(self.embedding_layer(x)))
        h = self.vit(self.embedding_layer(x))
        return h

    def forward_cluster(self, x, return_h=False):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        h = self.vit(self.embedding_layer(x))
        pred = self.clustering_head(h)
        labels = torch.argmax(pred, dim=1)
        if return_h:
            return labels, h
        return labels

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z)

        sim = torch.matmul(z, z.T) / self.temperature  # Dot similarity
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class CrossCorrelationLoss(nn.Module):

    def __init__(self, out_dim, lambd, device):
        super(CrossCorrelationLoss, self).__init__()
        self.lambd = lambd
        self.device = device
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, y_i, y_j):
        batch_size = y_i.size(0)
        c = self.bn(y_i).T @ self.bn(y_j)
        # sum the cross-correlation matrix between all gpus
        c = c / batch_size
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class ClusteringLoss(nn.Module):
    def __init__(self, weight_clu_loss=0.01, regularization_coef=0.05):
        super(ClusteringLoss, self).__init__()
        self.kl_criterion = nn.KLDivLoss(reduction='sum')
        self.regularization_coef = regularization_coef
        self.weight_clu_loss = weight_clu_loss

    def forward(self, y_prob, cluster_center=None):
        """
        :param y_prob: prob of embeddings
        :return:
        """
        target_prob = self.target_distribution(y_prob)  # .detach()
        loss = self.kl_criterion(y_prob.log(), target_prob)/y_prob.shape[0]
        reg_loss = 0.
        if cluster_center is not None:  # #  orthogonal regularization on centers: matmul(C, C^T) - I
            # cluster_center = F.normalize(cluster_center)
            x = torch.matmul(cluster_center, cluster_center.t())
            n, m = x.shape
            reg_loss = torch.norm(x - torch.eye(n).to(cluster_center.device)).pow(2).sum()
            # off_diag = x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().pow(2).sum()
            # loss += 1e-5 * reg_loss
        prob = y_prob.sum(0).view(-1)
        prob = prob / prob.sum()
        entropy = math.log(prob.size(0)) + (prob * torch.log(prob)).sum()
        loss = self.weight_clu_loss * loss + self.regularization_coef * (entropy + reg_loss)
        # loss += 1e-5 * reg_loss
        return loss

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-8)
        return (weight.t() / torch.sum(weight, 1)).t()

class PretrainLoss(nn.Module):
    """
    pretrain model for n epoch with a contrastive task, e.g., SimCLR/BarlowTwins
    """
    def __init__(self, batch_size, lambda_, device='cpu'):
        super(PretrainLoss, self).__init__()
        self.device = device
        self.criterion = InstanceLoss(batch_size, lambda_, device).to(device)
        # self.criterion = CrossCorrelationLoss(batch_size, lambda_, device).to(device)

    def forward(self, x_1, x_2):
        loss = self.criterion(x_1, x_2)
        return loss

class JointLoss(nn.Module):
    """
    joint train model with a center-based loss plussed with a contrastive loss
    """
    def __init__(self, batch_size, lambda_=0.5, weight_clu=1, regularization_coef=0.05, device='cpu'):
        super(JointLoss, self).__init__()
        self.device = device
        self.weight_clu = weight_clu
        self.regularization_coef = regularization_coef
        self.criterion_contrastive = InstanceLoss(batch_size, lambda_, device).to(device)
        self.clustering_loss = ClusteringLoss(weight_clu, regularization_coef)

    def forward(self, y_1, y_2, cluster_center=None):
        h = torch.cat([y_1, y_2], dim=0)
        loss_con = self.criterion_contrastive(y_1, y_2)
        loss_clu = self.clustering_loss(h, cluster_center)
        loss = loss_con + loss_clu
        return loss, loss_con, loss_clu


def class_acc(y_true, y_pre):
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]
        y_c_p = y_pre[np.nonzero(y_true == c)]
        ca.append(metrics.accuracy_score(y_c, y_c_p))
    return np.array(ca)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_accuracy(y_true, y_pre, return_aligned=False):
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = (y_true == Label1[i]).astype(float)
        for j in range(nClass2):
            ind_cla2 = (y_pre == Label2[j]).astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    row_ind, col_ind = linear_sum_assignment(-G.T)
    order = np.argsort(row_ind)
    c = col_ind[order]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / y_true.shape[0]
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model_path, model, optimizer, current_epoch):
    os.makedirs(model_path, exist_ok=True)
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


DATASET_LAYOUT = {
    'Augsburg': ('augsburg_gt.mat', 'augsburg_hsi.mat', 'augsburg_sar.mat'),
    'Houston':  ('gt.mat',          'HSI.mat',          'LiDAR.mat'),
    'MUUFL':    ('gt.mat',          'HSI.mat',          'LiDAR.mat'),
    'Trento':   ('gt.mat',          'HSI.mat',          'LiDAR.mat'),
}


def resolve_dataset_paths(dataset_root, dataset_name):
    if dataset_name not in DATASET_LAYOUT:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}. Choose from {list(DATASET_LAYOUT)}")
    root = Path(dataset_root) / dataset_name
    gt_name, m1_name, m2_name = DATASET_LAYOUT[dataset_name]
    gt_path, m1_path, m2_path = root / gt_name, root / m1_name, root / m2_name
    missing = [str(p) for p in (gt_path, m1_path, m2_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset files:\n  " + "\n  ".join(missing))
    return str(gt_path), (str(m1_path), str(m2_path))


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'1','true','yes','y','on'}:
        return True
    if s in {'0','false','no','n','off'}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


ORIGINAL_CONFIG = {
    'seed': 42,
    'workers': 2,
    'dataset': 'MUUFL',
    'dataset_root': 'dataset',
    'model_path': 'save/MUUFL',
    'batch_size': 256,
    'image_size': 7,
    'joint_train_epoch': 20,
    'dim_emebeding': 512,
    'lr_scale': 10,
    'is_labeled_pixel': False,
    'is_pretrain': False,
    'learning_rate': 0.0002,
    'weight_decay': 0.0005,
    'contrastive_param': 0.5,
    'weight_clu_loss': 1,
    'regularizer_coef': 0.05,
}

def train(model, loss_op, train_loader, optimizer):
    model.train()
    loss_epoch = 0
    for step, ((x_1, x_2), y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list_1 = [x_i.to(DEVICE) for x_i in x_1]
        x_list_2 = [x_i.to(DEVICE) for x_i in x_2]
        y1, y2 = model(x_list_1, x_list_2)
        loss_, loss_con, loss_clu = loss_op(y1, y2, model.clustering_head.cluster_centers)
        # loss_, loss_con, loss_clu = loss_op(y1, y2)

        loss_.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t loss: "  f"{loss_.item():.6f}\t" f'CL:{loss_con.item():.6f}\t CLU: {loss_clu.item():.6f}')
        loss_epoch += loss_.item()
    return loss_epoch

def inference(test_loader, model, device, is_labeled_pixel):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(test_loader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            pred = model.forward_cluster(x_list)
        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    y_pred_vector = np.array(y_pred_vector)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(y_pred_vector.shape))
    if is_labeled_pixel:
        acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels_vector, y_pred_vector)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y = labels_vector[indx_labeled]
        y_pred = y_pred_vector[indx_labeled]
        acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y, y_pred)
    print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}'.format(acc, kappa, nmi, ari, pur))
    return acc, kappa, nmi, ari, pur, ca


def build_parser():
    p = argparse.ArgumentParser(description='TMPCC' + ' standalone integration')
    for k, v in ORIGINAL_CONFIG.items():
        t = str2bool if isinstance(v, bool) else type(v)
        p.add_argument(f'--{k}', default=v, type=t)
    return p


def main():
    args = build_parser().parse_args()
    pretrain_path = os.path.join(args.model_path, 'pretrain')
    joint_train_path = os.path.join(args.model_path, 'joint-train')
    os.makedirs(pretrain_path, exist_ok=True)
    os.makedirs(joint_train_path, exist_ok=True)
    set_global_random_seed(args.seed)

    gt_path, img_path = resolve_dataset_paths(args.dataset_root, args.dataset)
    dataset_train = MultiModalDataset(
        gt_path, *img_path,
        patch_size=(args.image_size, args.image_size),
        transform=Transforms(size=args.image_size), is_labeled=False,
    )
    class_num = dataset_train.n_classes
    print('Processing %s ' % img_path[0])
    print(dataset_train.data_size, class_num)
    print(args)

    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, prefetch_factor=4,
    )
    dataset_test = MultiModalDataset(
        gt_path, *img_path,
        patch_size=(args.image_size, args.image_size),
        transform=None, is_labeled=args.is_labeled_pixel,
    )
    data_loader_test = DataLoader(
        dataset_test, batch_size=512, shuffle=False, drop_last=False,
        num_workers=args.workers,
    )

    model = Net(dataset_train.n_modality, dataset_train.in_channels, (args.image_size, args.image_size), 32, class_num, args.dim_emebeding).to(DEVICE)
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'clustering_head' not in n],
          'lr': args.learning_rate},
        {'params': model.clustering_head.cluster_centers, 'lr': args.learning_rate * args.lr_scale},
    ]
    optimizer = torch.optim.Adam(grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    score_list = []
    each_class = []
    acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, args.is_labeled_pixel)
    score_list.append([acc, kappa, nmi, ari, pur])
    print(f'initial accuracy: ACC={acc:.4f}')

    loss_op_joint = JointLoss(
        args.batch_size, lambda_=args.contrastive_param,
        weight_clu=args.weight_clu_loss,
        regularization_coef=args.regularizer_coef, device=DEVICE,
    )
    loss_history = []
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print('start fine-tuning ...')
    start_time = time.time()
    for epoch in range(1, args.joint_train_epoch + 1):
        loss_epoch = train(model, loss_op_joint, data_loader_train, optimizer)
        print(f"Epoch [{epoch}/{args.joint_train_epoch}]	 Loss: {loss_epoch / len(data_loader_train)}")
        acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, args.is_labeled_pixel)
        score_list.append([acc, kappa, nmi, ari, pur])
        each_class.append([ca])
        loss_history.append(loss_epoch / len(data_loader_train))
        lr_scheduler.step()
    running_time = time.time() - start_time
    print(f'fine tuning time: {running_time:.3f} s')
    save_model(joint_train_path, model, optimizer, args.joint_train_epoch)
    print(loss_history)
    print(score_list)
    if each_class:
        print(each_class)


if __name__ == '__main__':
    main()
