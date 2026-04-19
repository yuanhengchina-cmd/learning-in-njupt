import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from torch.nn import Parameter


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class FreqAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in (self.W_Q, self.W_K, self.W_V, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.W_Q(x).view(B, S, H, d).transpose(1, 2)
        k = self.W_K(x).view(B, S, H, d).transpose(1, 2)
        v = self.W_V(x).view(B, S, H, d).transpose(1, 2)

        q_fft = torch.fft.rfft(q, dim=2, norm="ortho")
        k_fft = torch.fft.rfft(k, dim=2, norm="ortho")
        v_fft = torch.fft.rfft(v, dim=2, norm="ortho")

        attn = q_fft * torch.conj(k_fft)
        attn = torch.softmax(attn.abs(), dim=2)
        attn = attn * torch.exp(1j * attn.angle().detach())

        out_fft = attn * v_fft
        out = torch.fft.irfft(out_fft, n=S, dim=2, norm="ortho")
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class transformerblock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = FreqAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)  # 严格保留，哪怕前向传播没用

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.residual_scale = Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + self.residual_scale * attn_out

        # 修复：严格遵循原代码逻辑，此处的 x 直接进入 mlp，不经过 norm2！
        mlp_out = self.mlp(x)
        x = x + self.residual_scale * mlp_out
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_sizes, embed_dim, dropout=0.1):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=size, stride=size)
            for size in patch_sizes
        ])
        for proj in self.projs:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        multi_scale_features = []
        spatial_shapes = []
        for proj in self.projs:
            x_proj = proj(x)
            multi_scale_features.append(x_proj.flatten(2).transpose(1, 2))
            spatial_shapes.append((x_proj.shape[2], x_proj.shape[3]))

        max_seq_len = max([feat.shape[1] for feat in multi_scale_features])
        unified_features = []
        for feat in multi_scale_features:
            if feat.shape[1] < max_seq_len:
                feat_resized = F.interpolate(feat.transpose(1, 2), size=max_seq_len, mode='nearest').transpose(1, 2)
                unified_features.append(feat_resized)
            else:
                unified_features.append(feat)

        x_fused = torch.stack(unified_features, dim=1).mean(dim=1)
        return self.dropout(self.norm(x_fused)), spatial_shapes


class ResG(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv2(self.relu(self.conv1(x)))
        return out + residual


class CAB(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        return self.fc(self.avg_pool(x).view(B, C)).view(B, C, 1, 1)


class MDFM(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_branch = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True))
        self.freq_branch = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
                                         nn.ReLU(inplace=True))
        self.concat_conv1x1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.resg = ResG(in_channels)
        self.post_conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cab = CAB(in_channels)

    def forward(self, F_s, F_l):
        X_s, X_f = self.spatial_branch(F_s), self.freq_branch(F_l)
        concat_feat = torch.cat([X_s, X_f], dim=1)
        x = self.concat_conv1x1(concat_feat)
        x = self.resg(x)
        x = self.post_conv3x3(x)
        W = self.cab(x)
        return X_f * W + X_s * (1 - W)


class FastFourierAdjustmentBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.amplitude_branch = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels, 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.hidden_channels, self.in_channels, 1))
        self.phase_branch = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels, 1), nn.ReLU(inplace=True),
                                          nn.Conv2d(self.hidden_channels, self.in_channels, 1))
        self.final_conv = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)

        # 修复：加回最初始化的权重，这对频域学习至关重要
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        residual = x
        x_fft = fft2(x, dim=(-2, -1))
        amplitude_opt = self.amplitude_branch(torch.abs(x_fft))
        phase_opt = self.phase_branch(torch.angle(x_fft).detach())
        x_fft_opt = amplitude_opt * torch.exp(1j * phase_opt)
        x_ifft = ifft2(x_fft_opt, dim=(-2, -1)).real
        return residual + self.final_conv(x_ifft)


class vit(nn.Module):
    def __init__(self, in_channels, patch_sizes, embed_dim, num_heads, num_layers, mlp_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mdfm = MDFM(in_channels)
        self.patch_embed = PatchEmbed(in_channels, patch_sizes, embed_dim, dropout)
        self.pos_embed = None

        mlp_dim = mlp_dim if mlp_dim is not None else embed_dim * 4
        self.blocks = nn.ModuleList(
            [transformerblock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.ffab = FastFourierAdjustmentBlock(in_channels)

    def _generate_pos_embed(self, seq_len, embed_dim, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-math.log(10000.0) / embed_dim))
        pos_embed = torch.zeros(1, seq_len, embed_dim, device=device)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed

    def forward(self, x):
        x_s = x
        x_f = torch.abs(torch.fft.fft2(x))
        x = self.mdfm(x_s, x_f)
        x = self.ffab(x)
        x, _ = self.patch_embed(x)

        S = x.shape[1]
        if self.pos_embed is None or self.pos_embed.shape[1] != S:
            self.pos_embed = self._generate_pos_embed(S, self.embed_dim, x.device)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.global_pool(self.norm(x).transpose(1, 2)).squeeze(-1)


class AutoEncoder(nn.Module):
    def __init__(self, n_input, window_size=16, hidden_channels=64, latent_dim=16):
        super().__init__()
        self.decoder_init_size = window_size
        patch_sizes = [window_size, max(1, window_size // 2), max(1, window_size // 4)]
        self.freq_encoder = vit(n_input, patch_sizes, hidden_channels, 8, 3, hidden_channels, 0.1)

        self.encoder_fc = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, latent_dim)
        )
        # 修复：原版代码强制使用 xavier 赋予更好的聚类起点
        nn.init.xavier_uniform_(self.encoder_fc[1].weight)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * window_size * window_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        # 修复：原版代码强制使用 xavier
        nn.init.xavier_uniform_(self.decoder_fc[0].weight)

        self.decoder_conv = nn.Sequential(
            View((-1, hidden_channels, window_size, window_size)),
            nn.Upsample(size=(window_size, window_size), mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1), nn.BatchNorm2d(hidden_channels // 2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, 3, padding=1), nn.BatchNorm2d(hidden_channels // 4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden_channels // 4, n_input, 3, padding=1), nn.Sigmoid()
        )

        # 修复：原版代码对解码器卷积的严苛初始化
        for m in self.decoder_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder_fc(self.freq_encoder(x))

    def decode(self, z):
        return self.decoder_conv(self.decoder_fc(z))

    def forward(self, x):
        return self.decode(self.encode(x))


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden, cluster_centers=None, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(n_clusters, hidden, dtype=torch.float).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha)) ** ((self.alpha + 1) / 2)
        return (numerator.t() / (torch.sum(numerator, 1) + 1e-8)).t()


class DEC(nn.Module):
    def __init__(self, n_clusters, autoencoder, hidden=10, alpha=1.0):
        super().__init__()
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(n_clusters, hidden, None, alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        return self.clusteringlayer(self.autoencoder.encode(x))