import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


def setup_seed(seed):
    import torch
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loadData(name):
    data_path = 'dataset/'
    if name == 'indian':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'salinas':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'ksc':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    elif name == 'muufl':
        data = sio.loadmat(os.path.join(data_path, 'Muufl.mat'))['HSI']
        labels = sio.loadmat(os.path.join(data_path, 'Muufl_gt.mat'))['gt']
    elif name == 'vnir':
        data = sio.loadmat(os.path.join(data_path, 'VNIR_corrected_2.mat'))['VNIR_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'VNIR_gt_2.mat'))['VNIR_gt']

        filter_mask = (labels == 3) | (labels == 4) | (labels == 6) | (labels == 7) | \
                      (labels == 8) | (labels == 11) | (labels == 15)
        labels[filter_mask] = 0

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        label_map = {old_val: i + 1 for i, old_val in enumerate(unique_labels)}
        new_labels = np.zeros_like(labels)
        for old_val, new_val in label_map.items():
            new_labels[labels == old_val] = new_val
        labels = new_labels
    else:
        raise ValueError("Unsupported dataset.")
    return data, labels


def infoChange(X, numComponents):
    X_copy = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    half = int(numComponents / 2)
    for i in range(0, half - 1):
        X_copy[:, :, i] = X[:, :, (half - i) * 2 - 1]
    for i in range(half, numComponents):
        X_copy[:, :, i] = X[:, :, (i - half) * 2]
    return X_copy


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    newX = infoChange(newX, numComponents)
    return newX


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def standardize_label(y):
    import copy
    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


def createImageCubes(X, y, windowSize=5, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1

    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    patchesLabels = standardize_label(patchesLabels)
    return patchesData, patchesLabels


def data_load(datasetname, patch_size=3, pca_components=20):
    setup_seed(42)
    X, y = loadData(datasetname)
    height, width = X.shape[0], X.shape[1]
    max_v, min_v = np.max(X, axis=(0, 1)), np.min(X, axis=(0, 1))
    X = (X - min_v) / (max_v - min_v)
    X_pca = applyPCA(X, numComponents=pca_components)
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    X = X.transpose(0, 3, 1, 2)
    print(f'Final sample shape: {X.shape}, labels: {np.unique(y)}')
    return X, y, height, width