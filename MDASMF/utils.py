import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score as nmi_score, \
    adjusted_mutual_info_score as ami_score, adjusted_rand_score as ari_score, \
    fowlkes_mallows_score as fmi_score, homogeneity_score, completeness_score, \
    v_measure_score, cohen_kappa_score as kappa
from scipy.optimize import linear_sum_assignment
from munkres import Munkres
import spectral as spy


def eva(y_true, y_pred, epoch, arg):
    non_background_indices = np.where(y_true != 0)
    label = y_true[non_background_indices]
    pred = y_pred[non_background_indices]

    acc, f1 = cluster_acc(label, pred)
    ari = ari_score(label, pred)
    ami = ami_score(label, pred)
    nmi = nmi_score(label, pred)
    homogeneity = homogeneity_score(label, pred)
    completeness = completeness_score(label, pred)
    Vmeasure = v_measure_score(label, pred)
    fmi = fmi_score(label, pred)

    # 仅在非 'pre' 阶段打印
    if epoch != 'pre':
        print(f'{epoch} ACC = {acc:.4f} ARI = {ari:.4f} AMI = {ami:.4f} NMI = {nmi:.4f}')
    return acc


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1, l2 = list(set(y_true)), list(set(y_pred))

    ind = 0
    if len(l1) > len(l2):
        for i in l1:
            if i not in l2:
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1
    elif len(l1) < len(l2):
        for i in l2:
            if i not in l1:
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    l1, l2 = list(set(y_true)), list(set(y_pred))
    cost = np.zeros((len(l1), len(l2)), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    indexes = Munkres().compute(cost.__neg__().tolist())
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    y_true = y_true[:len(y_true) - ind]
    new_predict = new_predict[:len(y_pred) - ind]

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def best_map(L1, L2):
    L1, L2 = np.array(L1), np.array(L2)
    Label1, Label2 = np.unique(L1), np.unique(L2)
    nClass = np.maximum(len(Label1), len(Label2))
    G = np.zeros((nClass, nClass))
    for i in range(len(Label1)):
        ind_cla1 = (L1 == Label1[i]).astype(int)
        for j in range(len(Label2)):
            ind_cla2 = (L2 == Label2[j]).astype(int)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    index = np.array(Munkres().compute(-G.T))
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(len(Label2)):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2.astype(int)


def paua(predicted, gt, cfg):
    # 补回安全检查：如果是 Tensor，先转成 NumPy 数组
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()

    predicted, gt = predicted.astype(int), gt.astype(int)
    class_counts = np.bincount(gt)
    true_positive = np.zeros(len(class_counts))
    for c in range(len(class_counts)):
        true_positive[c] = np.sum((predicted == c) & (gt == c))
    pred_counts = np.bincount(predicted, minlength=len(class_counts))

    pa = true_positive / (class_counts + 1e-8)
    ua = true_positive / (pred_counts + 1e-8)

    print(f"========== {cfg.name} PA & UA ==========")
    for c in np.unique(gt):
        print(f"Class {c:2d} | PA: {pa[c]:.4f} | UA: {ua[c]:.4f}")


def Draw_Classification_Map(label, name: str, dataset_name: str = None, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    v = spy.imshow(classes=np.array(label).astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gcf().savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)


def cluster_ac(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    y_true, y_pred = y_true - torch.min(y_true), y_pred - torch.min(y_pred)
    l1, l2 = list(set(y_true.tolist())), list(set(y_pred.tolist()))

    cost = torch.zeros((len(l1), len(l2)), dtype=torch.int32)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    row_ind, col_ind = linear_sum_assignment(-cost.numpy())
    new_predict = torch.zeros(len(y_pred), dtype=y_pred.dtype)
    mapping = {}
    for i, c in enumerate(l1):
        c2 = l2[col_ind[i]]
        new_predict[y_pred == c2] = c
        mapping[int(c2)] = int(c)

    y_true_np, new_predict_np = y_true.numpy(), new_predict.numpy()
    return metrics.accuracy_score(y_true_np, new_predict_np), new_predict, y_true, mapping, \
        None, kappa(y_true_np, new_predict_np), nmi_score(y_true_np, new_predict_np), \
        ari_score(y_true_np, new_predict_np), ami_score(y_true_np, new_predict_np), \
        fmi_score(y_true_np, new_predict_np), homogeneity_score(y_true_np, new_predict_np), \
        completeness_score(y_true_np, new_predict_np), v_measure_score(y_true_np, new_predict_np)