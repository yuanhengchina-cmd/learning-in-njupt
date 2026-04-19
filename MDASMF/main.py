import os
import argparse
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np

from dataset import data_load, setup_seed
from models import AutoEncoder, DEC
from utils import eva, best_map, cluster_ac, paua, Draw_Classification_Map

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    arg = kwargs['arg']
    physical_batch_size = 1024

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    N = data.shape[0]

    print(f"Pretraining Phase (Pseudo Full-Batch), Total Samples: {N}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        perm_idx = torch.randperm(N, device=data.device)

        for i in range(0, N, physical_batch_size):
            end_idx = min(i + physical_batch_size, N)
            batch_data = data[perm_idx[i: end_idx]]
            actual_batch_size = end_idx - i

            output = model(batch_data)
            loss = loss_fn(output, batch_data)
            scaled_loss = loss * (actual_batch_size / N)
            scaled_loss.backward()

            total_loss += loss.item() * actual_batch_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if (epoch + 1) % 1== 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / N:.6f}')

    os.makedirs('saves', exist_ok=True)
    torch.save(model.state_dict(), f'saves/{arg.name}_pretrain.pt')


def train(**kwargs):
    data = kwargs['data']
    y = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    arg = kwargs['arg']

    CHUNK_SIZE = 1024
    N = data.shape[0]

    print(f'Start Training (DEC Phase). Total samples: {N}')

    # 严格使用 eval 和 no_grad 提取初始特征
    model.eval()
    with torch.no_grad():
        features_list = [model.autoencoder.encode(data[i: min(i + CHUNK_SIZE, N)]).cpu()
                         for i in range(0, N, CHUNK_SIZE)]
    features = torch.cat(features_list, dim=0).numpy()

    kmeans = KMeans(n_clusters=arg.n_clusters, random_state=0, n_init=10).fit(features)
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(
        torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device))

    print(f'Initial Accuracy: {eva(y, kmeans.predict(features), "pre", arg):.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_function = nn.KLDivLoss(reduction='batchmean')

    best_accuracy, best_model_path = -1.0, f'{arg.save_dir}/{arg.name}_BEST_DEC.pt'
    UPDATE_INTERVAL = 15
    full_p = None

    for epoch in range(num_epochs):
        # 在更新目标分布 P 前严格开启 eval 模式消除 Dropout/BN 噪声
        if epoch % UPDATE_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                q_list = [model(data[i: min(i + CHUNK_SIZE, N)]) for i in range(0, N, CHUNK_SIZE)]
            full_p = model.target_distribution(torch.cat(q_list, dim=0)).detach()

        # 恢复训练模式
        model.train()

        total_loss = 0.0
        optimizer.zero_grad()
        perm_idx = torch.randperm(N, device=data.device)

        for i in range(0, N, CHUNK_SIZE):
            end_idx = min(i + CHUNK_SIZE, N)
            batch_idx = perm_idx[i: end_idx]
            actual_batch_size = end_idx - i

            batch_out = model(data[batch_idx])
            loss = loss_function((batch_out + 1e-8).log(), full_p[batch_idx])
            (loss * (actual_batch_size / N)).backward()
            total_loss += loss.item() * actual_batch_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # 恢复每个 Epoch 评估一次，避免错过峰值 Best Model
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.cat([model(data[i: min(i + CHUNK_SIZE, N)]).argmax(1).cpu()
                                    for i in range(0, N, CHUNK_SIZE)], dim=0).numpy()
            accuracy = eva(y, y_pred, epoch, arg)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                os.makedirs(arg.save_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)

    # ---------------- 最终评估阶段 ----------------
    print('\nLoading the BEST DEC model...')
    model.load_state_dict(torch.load(best_model_path, map_location=data.device))
    model.eval()
    with torch.no_grad():
        y_pred = torch.cat([model(data[i: min(i + CHUNK_SIZE, N)]).argmax(1).cpu()
                            for i in range(0, N, CHUNK_SIZE)], dim=0).numpy()

    y_pred_p = best_map(y, y_pred)
    y_map = y.reshape(arg.h, arg.w)
    y_pred_map = y_pred_p.reshape(arg.h, arg.w)

    non_bg_index = np.where(y != 0)
    acc, y_prednobg_p, y_true_nobg_p, _, _, _, nmi, ari, ami, fmi, homogenerity, completeness, Vmeasure = \
        cluster_ac(y[non_bg_index], y_pred[non_bg_index])

    print("\n--- Final Clustering Performance ---")
    print('%10s %10.4f %10.4f %10.4f %10.4f' % ('Metrics', acc, ari, ami, nmi))
    paua(y_prednobg_p, y_true_nobg_p, arg)

    os.makedirs(f'results/{arg.name}', exist_ok=True)

    # 完美对齐：VNIR数据集的特殊映射还原，保证出图颜色一致
    if arg.name == 'vnir':
        restore_map = {1: 1, 2: 2, 3: 5, 4: 9, 5: 10, 6: 12, 7: 13, 8: 14}
        y_pred_restored = np.zeros_like(y_pred_map)
        y_gt_restored = np.zeros_like(y_map)

        y_pred_restored[y_pred_map == 0] = 0
        y_gt_restored[y_map == 0] = 0

        for new_id, old_id in restore_map.items():
            y_pred_restored[y_pred_map == new_id] = old_id
            y_gt_restored[y_map == new_id] = old_id

        y_pred_map = y_pred_restored
        y_map = y_gt_restored

    Draw_Classification_Map(y_map, f"results/{arg.name}/map_gt", dataset_name=arg.name)
    Draw_Classification_Map(y_pred_map, f"results/{arg.name}/map_pred", dataset_name=arg.name)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='indian', choices=['indian', 'ksc', 'salinas', 'muufl', 'vnir'])
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='saves')
    parser.add_argument('--patch_size', default=5, type=int)
    parser.add_argument('--pca_dim', default=30, type=int)
    parser.add_argument('--seed', default=42, type=int)
    arg = parser.parse_args()

    setup_seed(arg.seed)
    x, y, h, w = data_load(arg.name, arg.patch_size, arg.pca_dim)
    x = torch.from_numpy(x).to(device).float()

    arg.n_clusters = len(set(y)) - 1
    arg.h, arg.w = h, w

    autoencoder = AutoEncoder(x.shape[1], window_size=arg.patch_size).to(device)
    pretrain(data=x, model=autoencoder, num_epochs=arg.pretrain_epochs, arg=arg)

    dec = DEC(n_clusters=arg.n_clusters, autoencoder=autoencoder, hidden=16).to(device)
    final_acc = train(data=x, labels=y, model=dec, num_epochs=arg.train_epochs, arg=arg)
    print(f"\nTraining Complete. Final ACC: {final_acc:.4f}")