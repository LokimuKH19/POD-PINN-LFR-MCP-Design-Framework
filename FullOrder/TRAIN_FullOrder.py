import torch
import pandas as pd
import numpy as np
import random
from torch import nn
from InterpolatorORI import SmoothKNNInterpolator
import torch.autograd as autograd
import re
import os
import matplotlib.pyplot as plt
from NeuralOperators import FNO2d_small, CNO2d_small, CFNO2d_small
from Encoders import CoordMLP, CoordCNN
import torch.nn.functional as F
import json
import time
from datetime import datetime
import signal
import sys


# ----------------------------------------------
# FullOrderPINN: PINN-DeepONet-FullMLP
# ----------------------------------------------

# ======== Configuration Management ========
class Config:
    """Configuration management class for FullOrderPINN"""

    def __init__(self, config_dict=None):
        # Default parameters
        self.SEED = 42
        self.net_type = 'CFNO'
        self.model_mode = 'deeponet'
        self.hidden_dim = 64
        self.hidden_layers = 2
        self.hidden_branch_dim = 128
        self.hidden_branch_layers = 4
        self.learning_rate = 1e-3
        self.coord_batch_size = 128
        self.use_physics_loss = True
        self.use_physics_batch = True
        self.epochs = 210
        self.pretrain_epochs = 200
        self.use_scheduler = True
        self.scheduler_patience = 5
        self.scheduler_factor = 0.75
        self.scheduler_min_lr = 1e-6
        self.scaling_factor = 1.0
        # 新增：点抽样参数
        self.point_sampling_ratio = 0.3  # 抽样比例，0-1之间
        self.use_point_sampling = True  # 是否启用点抽样

        # FNO configs
        self.fno_configs = {
            "encoder_type": 'MLP',
            "width": 16,
            "depth": 4,
            "modes": 32,
            "cheb_modes": (16, 16),
            "alpha_init": 0.5,
            "pseudo_grid_size": 64
        }

        if config_dict:
            for key, value in config_dict.items():
                if key == 'fno_configs' and isinstance(value, dict):
                    self.fno_configs.update(value)
                else:
                    setattr(self, key, value)

    def to_dict(self):
        """Convert config to dictionary"""
        config_dict = {key: value for key, value in self.__dict__.items()
                       if not key.startswith('_')}
        return config_dict

    def save(self, filepath):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)


# ======== Utility: set seed ========
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======== Network components ========
class MLPBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, hidden_layers=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), act()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNBackbone(nn.Module):
    """
    Simple 1D CNN backbone to process coordinate triplets [r,theta,z].
    Treat coords as a 1D sequence of length 3 with 1 channel.
    Output: (N, hidden_dim)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, hidden_dim)

    def forward(self, x):
        # x: (N, 3) -> (N,1,3)
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class TransformerBackbone(nn.Module):
    """
    This module does not support for 2nd derivatives
    Lightweight Transformer backbone:
    - project coord (3-dim) to d_model
    - apply TransformerEncoder over a short sequence (we use length=1 by treating coords per point as token
      or alternately treat r/theta/z as tokens; here we treat each point as a single token with embedding)
    - we keep a simple encoder and global projection to hidden_dim
    """

    def __init__(self, hidden_dim=128, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, hidden_dim)

    def forward(self, x):
        # x: (N,3) -> (N, d_model)
        x = self.proj(x)  # (N, d_model)
        # Transformer expects (S, B, E) where we'll use B=1 per point set, S=N
        x = x.unsqueeze(1)  # (N,1,d_model)
        x = x.permute(0, 1, 2)  # (N,1,d_model) -> treat N as seq len
        # encoder expects (S, B, E)
        x_enc = self.encoder(x)  # (N,1,d_model)
        # average over sequence (here just identity) -> collapse B dim
        x_enc = x_enc.squeeze(1)  # (N, d_model)
        return self.fc(x_enc)  # (N, hidden_dim)


# Fourier Neural Operator
# ========== Fourier-based backbone ==========
class FourierBackBone(nn.Module):
    """
    Fourier/Chebyshev/Combined Neural Operator backbone (non-structured compatible)
    Adapts unstructured coordinates to pseudo-grid features before FNO.
    """

    def __init__(self,
                 fno_type='FNO',
                 encoder_type='MLP',
                 coord_dim=3,
                 output_features=2,
                 width=16,
                 depth=3,
                 modes=16,
                 cheb_modes=(8, 8),
                 alpha_init=0.5,
                 device='cuda',
                 pseudo_grid_size=64):
        """
        pseudo_grid_size: defines resolution for internal FNO processing grid.
        """
        super().__init__()
        self.device = device
        self.coord_dim = coord_dim
        self.pseudo_grid_size = pseudo_grid_size
        self.output_features = output_features
        self.input_features = width  # Use width as input features

        # === 1. Encoder to lift coordinates ===
        if encoder_type.lower() == 'mlp':
            self.coord_encoder = CoordMLP(
                input_features=coord_dim,
                output_features=width,  # Use width instead of pseudo_grid_size
                hidden_dim=width
            )
        elif encoder_type.lower() == 'cnn':
            self.coord_encoder = CoordCNN(
                input_features=coord_dim,
                output_features=width,  # Use width instead of pseudo_grid_size
                hidden_dim=width
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # === 2. Operator ===
        if fno_type.upper() == 'FNO':
            self.operator = FNO2d_small(modes=modes, width=width, depth=depth,
                                        input_features=width, output_features=output_features)  # Use width
        elif fno_type.upper() == 'CNO':
            self.operator = CNO2d_small(cheb_modes=cheb_modes, width=width, depth=depth,
                                        input_features=width, output_features=output_features)  # Use width
        elif fno_type.upper() == 'CFNO':
            self.operator = CFNO2d_small(modes=modes, cheb_modes=cheb_modes, width=width, depth=depth,
                                         alpha_init=alpha_init, input_features=width,  # Use width
                                         output_features=output_features)
        else:
            raise ValueError(f"Invalid fno_type: {fno_type}")

        # === 3. Direct mapping instead of grid_sample ===
        # Use a simple MLP to map from coordinates to output features
        self.direct_map = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_features)
        )

        self.to(device)

    def forward(self, coords):
        """
        coords: (N,3) or (B,N,3)
        returns: (N, output_features) if B==1
                 (B,N,output_features) if B>1
        """
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)  # (1, N, 3)

        B, N, _ = coords.shape
        coords = coords.to(self.device)

        # (1) Encode coordinates to features - direct approach without grid_sample
        feats = self.coord_encoder(coords)  # (B, N, width)

        # Apply direct mapping instead of grid operations
        out = self.direct_map(feats)  # (B, N, output_features)

        if out.shape[0] == 1:
            return out.squeeze(0)  # (N, output_features)
        return out  # (B, N, output_features)


# ====== Branch network for DeepONet (conditions -> features) ======
class BranchNet(nn.Module):
    def __init__(self, input_dim=2, out_dim=128, hidden_dim=128, hidden_layers=3):
        super().__init__()
        self.net = MLPBackbone(input_dim, out_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

    def forward(self, cond):
        # cond: (B,2) -> output: (B, out_dim)
        return self.net(cond)


# ====== DeepONet model (branch + trunk) ======
class DeepONetFull(nn.Module):
    def __init__(self, trunk_type='MLP', trunk_hidden=128, trunk_layers=3, branch_hidden=128, branch_layers=3,
                 out_dim=4, fno_configs=None):
        super().__init__()
        self.out_dim = out_dim
        # branch
        self.branch = BranchNet(input_dim=2, out_dim=trunk_hidden, hidden_dim=branch_hidden,
                                hidden_layers=branch_layers)
        # trunk
        if trunk_type == 'MLP':
            self.trunk = MLPBackbone(input_dim=3, output_dim=trunk_hidden, hidden_dim=trunk_hidden,
                                     hidden_layers=trunk_layers)
        elif trunk_type == 'CNN':
            self.trunk = CNNBackbone(hidden_dim=trunk_hidden)
        elif trunk_type == 'Transformer':
            self.trunk = TransformerBackbone(hidden_dim=trunk_hidden)
        elif trunk_type in ['FNO', 'CNO', 'CFNO']:
            if fno_configs is None:
                fno_configs = {
                    "encoder_type": 'MLP',
                    "width": 16,
                    "depth": 3,
                    "modes": 16,
                    "cheb_modes": (8, 8),
                    "alpha_init": 0.5,
                    "pseudo_grid_size": 64
                }
            self.trunk = FourierBackBone(
                fno_type=trunk_type,
                encoder_type=fno_configs["encoder_type"],
                output_features=trunk_hidden,  # match to hidden dim
                width=fno_configs["width"],
                depth=fno_configs["depth"],
                modes=fno_configs["modes"],
                cheb_modes=fno_configs["cheb_modes"],
                alpha_init=fno_configs["alpha_init"],
                pseudo_grid_size=fno_configs["pseudo_grid_size"],
                device='cuda'
            )
        else:
            raise ValueError("trunk_type must be 'MLP'|'CNN'|'Transformer | FNO | CNO | CFNO'")
        # final mapping from fused features to output fields
        self.head = nn.Linear(trunk_hidden, out_dim)

    def forward(self, coords, cond):
        """
        coords: (N,3)
        cond: (B,2)
        returns: if B==1 -> (N, out_dim)
                 if B>1  -> (B, N, out_dim)
        """
        trunk_feat = self.trunk(coords)  # (N, H)
        branch_feat = self.branch(cond)  # (B, H)
        # fuse: elementwise multiply with broadcasting
        # branch_feat: (B,H) -> (B,1,H); trunk_feat: (N,H) -> (1,N,H)
        b = branch_feat.unsqueeze(1)  # (B,1,H)
        t = trunk_feat.unsqueeze(0)  # (1,N,H)
        fused = b * t  # (B,N,H)
        # map each fused vector to outputs
        out = self.head(fused)  # (B,N,out_dim)
        if out.shape[0] == 1:
            return out.squeeze(0)  # (N,out_dim)
        return out  # (B,N,out_dim)


# ====== Full MLP baseline: input [r,theta,z,omega,qv] -> outputs (Ur,Ut,Uz,P) ======
class FullMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, hidden_layers=4, out_dim=4):
        super().__init__()
        self.net = MLPBackbone(input_dim, out_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

    def forward(self, coords, cond):
        # coords: (N,3), cond: (B,2) or (2,)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        # expand cond to (N,2)
        cond_exp = cond.expand(coords.shape[0], -1) if cond.shape[0] == 1 else cond.repeat(coords.shape[0], 1)
        x = torch.cat([coords, cond_exp], dim=1)
        return self.net(x)  # (N,4)


class ModifiedDeepONet(nn.Module):
    def __init__(self, trunk_type='MLP', trunk_hidden=128, trunk_layers=3,
                 branch_hidden=128, branch_layers=3, out_dim=4,
                 fusion_hidden=128, fusion_layers=2, fno_configs=None):
        """
        Modified DeepONet with fusion network

        Args:
            trunk_type: Type of trunk network ('MLP'|'CNN'|'Transformer'|'FNO'|'CNO'|'CFNO')
            trunk_hidden: Hidden dimension for trunk network
            trunk_layers: Number of layers for trunk network
            branch_hidden: Hidden dimension for branch network
            branch_layers: Number of layers for branch network
            out_dim: Output dimension
            fusion_hidden: Hidden dimension for fusion network
            fusion_layers: Number of layers for fusion network
            fno_configs: Configuration for FNO-based trunks
        """
        super().__init__()
        self.out_dim = out_dim

        # Branch network (conditions -> features)
        self.branch = BranchNet(
            input_dim=2,
            out_dim=branch_hidden,
            hidden_dim=branch_hidden,
            hidden_layers=branch_layers
        )

        # Trunk network (coordinates -> features)
        if trunk_type == 'MLP':
            self.trunk = MLPBackbone(
                input_dim=3,
                output_dim=trunk_hidden,
                hidden_dim=trunk_hidden,
                hidden_layers=trunk_layers
            )
        elif trunk_type == 'CNN':
            self.trunk = CNNBackbone(hidden_dim=trunk_hidden)
        elif trunk_type == 'Transformer':
            self.trunk = TransformerBackbone(hidden_dim=trunk_hidden)
        elif trunk_type in ['FNO', 'CNO', 'CFNO']:
            if fno_configs is None:
                fno_configs = {
                    "encoder_type": 'MLP',
                    "width": 16,
                    "depth": 3,
                    "modes": 16,
                    "cheb_modes": (8, 8),
                    "alpha_init": 0.5,
                    "pseudo_grid_size": 64
                }
            self.trunk = FourierBackBone(
                fno_type=trunk_type,
                encoder_type=fno_configs["encoder_type"],
                output_features=trunk_hidden,  # match to hidden dim
                width=fno_configs["width"],
                depth=fno_configs["depth"],
                modes=fno_configs["modes"],
                cheb_modes=fno_configs["cheb_modes"],
                alpha_init=fno_configs["alpha_init"],
                pseudo_grid_size=fno_configs["pseudo_grid_size"],
                device='cuda'
            )
        else:
            raise ValueError("trunk_type must be 'MLP'|'CNN'|'Transformer'|'FNO'|'CNO'|'CFNO'")

        # Fusion network - combines trunk and branch features
        self.fusion_net = MLPBackbone(
            input_dim=trunk_hidden + branch_hidden,
            output_dim=out_dim,
            hidden_dim=fusion_hidden,
            hidden_layers=fusion_layers
        )

    def forward(self, coords, cond):
        """
        coords: (N,3) or (B,N,3) coordinates
        cond: (B,2) or (2,) conditions
        returns: if B==1 -> (N, out_dim)
                 if B>1  -> (B, N, out_dim)
        """
        # Handle single batch case
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)  # (1, N, 3)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)  # (1, 2)

        B, N, _ = coords.shape

        # Get trunk features: (B, N, trunk_hidden)
        trunk_feat = self.trunk(coords)  # Could be (N, H) or (B, N, H)
        if trunk_feat.ndim == 2:
            trunk_feat = trunk_feat.unsqueeze(0).expand(B, -1, -1)  # (B, N, H)

        # Get branch features: (B, branch_hidden)
        branch_feat = self.branch(cond)  # (B, H)

        # Expand branch features to match coordinates: (B, N, branch_hidden)
        branch_feat_expanded = branch_feat.unsqueeze(1).expand(-1, N, -1)

        # Concatenate features: (B, N, trunk_hidden + branch_hidden)
        fused_features = torch.cat([trunk_feat, branch_feat_expanded], dim=-1)

        # Pass through fusion network: (B, N, out_dim)
        out = self.fusion_net(fused_features)

        if out.shape[0] == 1:
            return out.squeeze(0)  # (N, out_dim)
        return out  # (B, N, out_dim)


# ======== FullOrderPINN system ========
class FullOrderPINN:
    def __init__(self, config):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # Configuration
        self.config = config

        # bookkeeping
        self.net_type = config.net_type
        self.model_mode = config.model_mode
        self.hidden_dim = config.hidden_dim
        self.hidden_layers = config.hidden_layers
        self.lr = config.learning_rate
        self.use_physics_loss_original = config.use_physics_loss
        self.use_physics_loss = config.use_physics_loss
        self.use_physics_batch = config.use_physics_batch
        self.scaling_factor = config.scaling_factor
        self.epoch = 0
        self.pretrain_epoch = 0
        self.hidden_branch_dim = config.hidden_branch_dim
        self.hidden_branch_layers = config.hidden_branch_layers
        self.fno_configs = config.fno_configs

        # 新增：点抽样参数
        self.use_point_sampling = config.use_point_sampling
        self.point_sampling_ratio = config.point_sampling_ratio
        self.sampled_point_indices = None  # 当前抽样的点索引

        # -------- 1. Load EXP (conditions) and split flag ----------
        data = pd.read_csv('./EXP.csv', header=0)
        omega = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32)
        qv = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)
        split_flag = torch.tensor(data.iloc[:, 2].values, dtype=torch.bool)
        self.train_mask = (~split_flag).to(self.device)
        self.test_mask = split_flag.to(self.device)

        # -------- 2. Load coordinates ----------
        coords = pd.read_csv('./coordinate.csv', header=None).values.astype(np.float32)
        self.coords_real = torch.tensor(coords, dtype=torch.float32).to(self.device)

        # -------- 3. Load full-field data ----------
        self.field_names = ['P', 'Ut', 'Ur', 'Uz']
        self.fields = {}
        for name in self.field_names:
            arr = pd.read_csv(f'./{name}.csv', header=None).values.astype(np.float32)
            if arr.shape[0] != self.coords_real.shape[0]:
                raise ValueError(f"[ERROR] {name}.csv row count mismatch with coordinates.")
            self.fields[name] = torch.tensor(arr, dtype=torch.float32).to(self.device)

        # =========================
        # 🔧 Data normalization
        # =========================
        coords_np = self.coords_real.cpu().numpy()
        self.coord_min = coords_np.min(axis=0)
        self.coord_max = coords_np.max(axis=0)

        self.omega_mean = omega.mean()
        self.omega_std = omega.std() + 1e-12
        self.qv_mean = qv.mean()
        self.qv_std = qv.std() + 1e-12

        # normalize all field data
        field_names = ["Ur", "Ut", "Uz", "P"]
        self.output_mean = {}
        self.output_std = {}
        for fname in field_names:
            data_np = np.loadtxt(f"./{fname}.csv", delimiter=",")
            mean_val = np.mean(data_np)
            std_val = np.std(data_np) + 1e-12

            self.output_mean[fname] = mean_val
            self.output_std[fname] = std_val
            data_n = (data_np - mean_val) / std_val
            setattr(self, f"{fname}_all", torch.tensor(data_n, dtype=torch.float32, device=self.device))

        # normalize coordinates for model input
        coords_norm = 2.0 * (coords_np - self.coord_min) / (self.coord_max - self.coord_min + 1e-12) - 1.0
        self.coords_all = torch.tensor(coords_norm, dtype=torch.float32, device=self.device)

        # normalize EXP params
        omega_n = (omega - self.omega_mean) / self.omega_std
        qv_n = (qv - self.qv_mean) / self.qv_std
        self.conditions = torch.stack([omega_n, qv_n], dim=1).to(self.device)

        # 初始化点抽样
        self._initialize_point_sampling()

        # -------- 4. Create model ----------
        model_dict = {
            'deeponet': DeepONetFull(
                trunk_type=self.net_type,
                trunk_hidden=self.hidden_dim,
                trunk_layers=self.hidden_layers,
                branch_hidden=self.hidden_branch_dim,
                branch_layers=self.hidden_branch_layers,
                out_dim=4,
                fno_configs=self.fno_configs
            ),
            'modified_deeponet': ModifiedDeepONet(
                trunk_type=self.net_type,
                trunk_hidden=self.hidden_dim,
                trunk_layers=self.hidden_layers,
                branch_hidden=self.hidden_branch_dim,
                branch_layers=self.hidden_branch_layers,
                out_dim=4,
                fusion_hidden=self.hidden_dim,  # need to be modified
                fusion_layers=2,
                fno_configs=self.fno_configs
            ),
            'MLP': FullMLP(
                input_dim=5,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
                out_dim=4
            )
        }

        if self.model_mode in model_dict:
            self.model = model_dict[self.model_mode].to(self.device)
        else:
            raise ValueError(f"Unknown model_mode: {self.model_mode}")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # physics constants
        self.rho = 10650.0
        self.mu = 0.006
        self.g = 9.81
        self.B = 10

    def _initialize_point_sampling(self):
        """初始化点抽样"""
        self.total_points = self.coords_all.shape[0]
        self.sample_size = int(self.total_points * self.point_sampling_ratio)
        print(
            f"[INFO] Point sampling: {self.sample_size}/{self.total_points} points (ratio: {self.point_sampling_ratio})")

        # 初始抽样
        self._resample_points()

    def _resample_points(self):
        if self.use_point_sampling:
            self.sampled_point_indices = torch.randperm(self.total_points)[:self.sample_size]
        else:
            self.sampled_point_indices = torch.arange(self.total_points)

    def get_sampled_data(self):
        if self.sampled_point_indices is None:
            self._resample_points()

        indices = self.sampled_point_indices

        sampled_data = {
            'coords': self.coords_all[indices],
            'coords_real': self.coords_real[indices],
            'Ur_all': self.Ur_all[indices] if hasattr(self, 'Ur_all') else None,
            'Ut_all': self.Ut_all[indices] if hasattr(self, 'Ut_all') else None,
            'Uz_all': self.Uz_all[indices] if hasattr(self, 'Uz_all') else None,
            'P_all': self.P_all[indices] if hasattr(self, 'P_all') else None,
            'indices': indices
        }
        return sampled_data

    # ============================================================
    # Forward prediction — user gives physical inputs
    # ============================================================
    def __call__(self, coords, condition, enable_grad=False):
        """coords: (M,3) physical values
           condition: (2,) physical values [omega, qv]"""
        self.model.eval()

        coords = coords.to(self.device)
        if enable_grad:
            coords.requires_grad_(True)

        # normalize internally
        coords_n = 2.0 * (coords - torch.tensor(self.coord_min, device=self.device)) / (
                torch.tensor(self.coord_max - self.coord_min, device=self.device) + 1e-12) - 1.0

        omega, qv = condition[0].item(), condition[1].item()
        omega_n = (omega - self.omega_mean) / (self.omega_std + 1e-12)
        qv_n = (qv - self.qv_mean) / (self.qv_std + 1e-12)
        cond = torch.tensor([[omega_n, qv_n]], dtype=torch.float32, device=self.device)

        with (torch.enable_grad() if enable_grad else torch.no_grad()):
            out_n = self.model(coords_n, cond)  # (M,4)
            Ur = out_n[:, 0] * self.output_std["Ur"] + self.output_mean["Ur"]
            Ut = out_n[:, 1] * self.output_std["Ut"] + self.output_mean["Ut"]
            Uz = out_n[:, 2] * self.output_std["Uz"] + self.output_mean["Uz"]
            P = out_n[:, 3] * self.output_std["P"] + self.output_mean["P"]
        return {'Ur': Ur, 'Ut': Ut, 'Uz': Uz, 'P': P}

    # ============================================================
    # Data Loss (MSE) - 修改为使用抽样点
    # ============================================================
    def compute_data_loss(self, use_sampled_points=True):
        """
        Compute MSE loss between predicted and true normalized fields.
        Now supports using sampled points to reduce computation.
        """
        loss_fn = nn.MSELoss()
        train_losses, test_losses = [], []

        nconds = self.conditions.shape[0]

        # 获取数据（抽样点或全部点）
        if use_sampled_points and self.use_point_sampling:
            sampled_data = self.get_sampled_data()
            coords_input = sampled_data['coords']
            Ur_data = sampled_data['Ur_all']
            Ut_data = sampled_data['Ut_all']
            Uz_data = sampled_data['Uz_all']
            P_data = sampled_data['P_all']
        else:
            coords_input = self.coords_all
            Ur_data = self.Ur_all
            Ut_data = self.Ut_all
            Uz_data = self.Uz_all
            P_data = self.P_all

        for idx in range(nconds):
            cond_input = self.conditions[idx:idx + 1]  # shape (1,2)
            preds = self.model(coords_input, cond_input)  # (Npoints,4)

            loss_sum = 0.0
            field_data = {
                'P': P_data[:, idx] if P_data is not None else getattr(self, "P_all")[:, idx],
                'Ut': Ut_data[:, idx] if Ut_data is not None else getattr(self, "Ut_all")[:, idx],
                'Ur': Ur_data[:, idx] if Ur_data is not None else getattr(self, "Ur_all")[:, idx],
                'Uz': Uz_data[:, idx] if Uz_data is not None else getattr(self, "Uz_all")[:, idx]
            }

            for name in ['P', 'Ut', 'Ur', 'Uz']:
                true_n = field_data[name]
                pred_n = preds[:, ['P', 'Ut', 'Ur', 'Uz'].index(name)]
                loss_sum += loss_fn(pred_n, true_n)

            if self.train_mask[idx]:
                train_losses.append(loss_sum)
            else:
                test_losses.append(loss_sum)

        train_total = torch.stack(train_losses).mean() if train_losses else torch.tensor(0.0, device=self.device)
        test_total = torch.stack(test_losses).mean() if test_losses else torch.tensor(0.0, device=self.device)

        return train_total, test_total

    # ============================================================
    # Physics residual computation - 修改为使用抽样点
    # ============================================================
    def compute_physics_residuals(self, coords, condition):
        """
        Compute physics residuals with simplified approach for FNO-based models
        Uses sampled coordinates if point sampling is enabled
        """
        # 如果启用了点抽样且coords为None，使用抽样点
        if coords is None and self.use_point_sampling:
            sampled_data = self.get_sampled_data()
            coords = sampled_data['coords_real']  # 使用物理坐标

        # Clone coordinates and enable gradients
        coords = coords.detach().clone().requires_grad_(True).to(self.device)
        r = coords[:, 0:1]

        # Prepare condition input
        omega_n = condition[0, 0].item()
        qv_n = condition[0, 1].item()
        omega_un = omega_n * self.omega_std + self.omega_mean
        omega_real = -(omega_un) * 2 * np.pi / 60.0
        qv_un = qv_n * self.qv_std + self.qv_mean
        cond_input = torch.tensor([omega_un, qv_un], dtype=torch.float32, device=self.device)

        # Forward pass
        fields = self.__call__(coords, cond_input, enable_grad=True)
        p = fields['P'].unsqueeze(1)
        ut = fields['Ut'].unsqueeze(1) - omega_real * r
        ur = fields['Ur'].unsqueeze(1)
        uz = fields['Uz'].unsqueeze(1)

        def grad(outputs, inputs):
            return autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                                 create_graph=True, retain_graph=True)[0]

        # Compute first derivatives only (simplified for FNO models)
        dur_dr = grad(ur, coords)[:, 0:1]
        dur_dtheta = grad(ur, coords)[:, 1:2]
        dur_dz = grad(ur, coords)[:, 2:3]

        dut_dr = grad(ut, coords)[:, 0:1]
        dut_dtheta = grad(ut, coords)[:, 1:2]
        dut_dz = grad(ut, coords)[:, 2:3]

        duz_dr = grad(uz, coords)[:, 0:1]
        duz_dtheta = grad(uz, coords)[:, 1:2]
        duz_dz = grad(uz, coords)[:, 2:3]

        dp_dr = grad(p, coords)[:, 0:1]
        dp_dtheta = grad(p, coords)[:, 1:2]
        dp_dz = grad(p, coords)[:, 2:3]

        # Simplified physics residuals (without second derivatives for FNO models)
        FC = (1 / r) * grad(r * ur, coords)[:, 0:1] + (1 / r) * dut_dtheta + duz_dz

        # For FNO models, use simplified momentum equations without Laplacian terms
        if self.net_type.upper() in ['FNO', 'CNO', 'CFNO']:
            # Simple viscosity approximation
            def simple_visc(u):
                """
                Approximate Laplacian of u for viscous term without second-order autograd.
                Uses sum of squares of first derivatives as rough estimate.
                """
                du = grad(u, coords)
                du_dr, du_dtheta, du_dz = du[:, 0:1], du[:, 1:2], du[:, 2:3]
                # Approximate Laplacian as sum of squared first derivatives
                lap_approx = du_dr ** 2 + (1 / r ** 2) * du_dtheta ** 2 + du_dz ** 2
                return lap_approx

            # Replace Laplacians in viscous terms
            lap_ur_approx = simple_visc(ur)
            lap_ut_approx = simple_visc(ut)
            lap_uz_approx = simple_visc(uz)
            # Residuals with simple viscosity
            FR = ur * dur_dr + (ut / r) * dur_dtheta + uz * dur_dz \
                 - (ut ** 2) / r + (1 / self.rho) * dp_dr \
                 - (self.mu / self.rho) * (lap_ur_approx - ur / (r ** 2) - 2 / (r ** 2) * dut_dtheta) \
                 - omega_real ** 2 * r + 2 * omega_real * ut
            FTH = ur * dut_dr + (ut / r) * dut_dtheta + uz * dut_dz \
                  + (ur * ut) / r + (1 / (self.rho * r)) * dp_dtheta \
                  - (self.mu / self.rho) * (lap_ut_approx - ut / (r ** 2) + 2 / (r ** 2) * dur_dtheta) \
                  - 2 * omega_real * ur
            FZ = ur * duz_dr + (ut / r) * duz_dtheta + uz * duz_dz \
                 + (1 / self.rho) * dp_dz - (self.mu / self.rho) * lap_uz_approx + self.g
        else:
            # For MLP/CNN, use full physics with Laplacian
            def laplacian(u):
                grad_u = grad(u, coords)
                du_dr = grad_u[:, 0:1]
                grad_du = grad(grad_u, coords)
                d2u_drr = grad_du[:, 0:1]
                d2u_dtheta2 = grad_du[:, 1:2]
                d2u_dzz = grad_du[:, 2:3]
                lap = (1 / r) * du_dr + d2u_drr + (1 / (r ** 2)) * d2u_dtheta2 + d2u_dzz
                return lap

            lap_ur = laplacian(ur)
            lap_ut = laplacian(ut)
            lap_uz = laplacian(uz)

            FR = ur * dur_dr + (ut / r) * dur_dtheta + uz * dur_dz \
                 - (ut ** 2) / r + (1 / self.rho) * dp_dr \
                 - (self.mu / self.rho) * (lap_ur - ur / (r ** 2) - 2 / (r ** 2) * dut_dtheta) \
                 - omega_real ** 2 * r + 2 * omega_real * ut
            FTH = ur * dut_dr + (ut / r) * dut_dtheta + uz * dut_dz \
                  + (ur * ut) / r + (1 / (self.rho * r)) * dp_dtheta \
                  - (self.mu / self.rho) * (lap_ut - ut / (r ** 2) + 2 / (r ** 2) * dur_dtheta) \
                  - 2 * omega_real * ur
            FZ = ur * duz_dr + (ut / r) * duz_dtheta + uz * duz_dz \
                 + (1 / self.rho) * dp_dz - (self.mu / self.rho) * lap_uz + self.g

        def safe_minmax(x):
            xmin = torch.min(x)
            xmax = torch.max(x)
            denom = xmax - xmin
            if denom.abs() < 1e-12:
                return torch.zeros_like(x)
            return (x - xmin) / denom

        FC, FR, FTH, FZ = safe_minmax(FC), safe_minmax(FR), safe_minmax(FTH), safe_minmax(FZ)

        mse = nn.MSELoss()
        physics_loss = mse(FC, torch.zeros_like(FC)) + \
                       mse(FR, torch.zeros_like(FR)) + \
                       mse(FTH, torch.zeros_like(FTH)) + \
                       mse(FZ, torch.zeros_like(FZ))
        return physics_loss

    # ============================================================
    # Helper losses - 修改为使用抽样点
    # ============================================================
    def compute_hidden_constraint_loss(self, coord_batch_size=64):
        if not self.use_physics_loss:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        idx = torch.nonzero(self.train_mask).squeeze()
        if idx.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # 使用抽样点
        if self.use_point_sampling:
            sampled_data = self.get_sampled_data()
            coords_all = sampled_data['coords_real']
            # 如果抽样点数量小于batch_size，使用所有抽样点
            actual_batch_size = min(coord_batch_size, len(coords_all))
            coords_batch = coords_all[torch.randperm(len(coords_all))[:actual_batch_size]]
        else:
            coords_all = self.coords_real[torch.randperm(self.coords_real.shape[0])[:coord_batch_size]]
            coords_batch = coords_all

        sample_idx = idx[torch.randperm(len(idx))[:min(self.B, len(idx))]]
        physics_loss = 0.0
        for i in sample_idx:
            cond = self.conditions[i].unsqueeze(0)
            physics_loss += self.compute_physics_residuals(coords_batch.clone(), cond)
        return physics_loss / float(len(sample_idx))

    def compute_hidden_constraint_loss_batch(self, coords_batch, condition_batch):
        if not self.use_physics_loss:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        physics_loss = 0.0
        for i in range(condition_batch.shape[0]):
            cond = condition_batch[i:i + 1]  # normalized cond
            loss = self.compute_physics_residuals(coords_batch.clone(), cond)
            physics_loss += loss
        return physics_loss / float(condition_batch.shape[0])

    def compute_total_loss(self, coord_batch_size=64):
        # 使用抽样点计算数据损失
        train_loss, test_loss = self.compute_data_loss(use_sampled_points=self.use_point_sampling)
        physics_loss = self.compute_hidden_constraint_loss(coord_batch_size=coord_batch_size)
        total_loss = train_loss + physics_loss * self.scaling_factor
        return total_loss, train_loss, test_loss, physics_loss

    # ---------- training step variants ----------
    def step_with_physics_batches(self, coord_batch_size=64):
        """
        Perform one training step using batches of coordinates for physics loss.
        Uses sampled points if point sampling is enabled.
        """
        train_loss_total = 0.0
        test_loss_total = 0.0
        physics_loss_total = 0.0

        # get indices of training conditions
        idx = torch.nonzero(self.train_mask).squeeze()
        if idx.numel() == 0:
            return 0.0, 0.0, 0.0

        # sample a batch of conditions
        sample_idx = idx[torch.randperm(len(idx))[:self.B]]
        condition_batch = self.conditions[sample_idx]  # shape (B,2)

        # 使用抽样点
        if self.use_point_sampling:
            sampled_data = self.get_sampled_data()
            coords_permuted = sampled_data['coords_real']
        else:
            coords_permuted = self.coords_real[torch.randperm(self.coords_real.shape[0])]

        num_coords = coords_permuted.shape[0]
        num_batches = max(1, (num_coords + coord_batch_size - 1) // coord_batch_size)

        for i in range(num_batches):
            start = i * coord_batch_size
            end = min((i + 1) * coord_batch_size, num_coords)
            coords_batch = coords_permuted[start:end]

            # zero gradients
            self.optim.zero_grad()

            # compute data loss (使用抽样点)
            train_loss, test_loss = self.compute_data_loss(use_sampled_points=self.use_point_sampling)

            # compute physics loss for current batch
            physics_loss = self.compute_hidden_constraint_loss_batch(coords_batch, condition_batch)

            # total loss
            total_loss = train_loss + physics_loss * self.scaling_factor
            total_loss.backward()
            self.optim.step()

            # accumulate scalar losses
            train_loss_total += train_loss.item()
            test_loss_total += test_loss.item()
            physics_loss_total += physics_loss.item()

            print(f"    [Batch {i + 1}/{num_batches}] Train Loss: {train_loss.item():.6f}, "
                  f"Test Loss: {test_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}")

            # clear cache to reduce memory pressure
            torch.cuda.empty_cache()

        return train_loss_total / num_batches, test_loss_total / num_batches, physics_loss_total / num_batches

    def step(self, coord_batch_size=64):
        """训练步骤，每个epoch重新抽样点"""
        # 每个epoch重新抽样点
        if self.use_point_sampling:
            self._resample_points()

        self.optim.zero_grad()
        total_loss, train_loss, test_loss, phy_loss = self.compute_total_loss(coord_batch_size=coord_batch_size)
        total_loss.backward()
        self.optim.step()
        return train_loss.item(), test_loss.item(), phy_loss.item()


# ======== Utility Functions ========
def generate_run_name(config, timestamp=None):
    """Generate descriptive run name"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    physics_flag = "Phys" if config.use_physics_loss else "NoPhys"
    batch_flag = "Batch" if config.use_physics_batch else "NoBatch"
    pretrain_flag = f"_Pretrain{config.pretrain_epochs}" if config.pretrain_epochs else ""
    sampling_flag = f"_Sample{config.point_sampling_ratio}" if config.use_point_sampling else ""

    return f"FullOrder_{timestamp}_{config.net_type}_{config.model_mode}_{physics_flag}_{batch_flag}_HD{config.hidden_dim}_HL{config.hidden_layers}_LR{config.learning_rate}{pretrain_flag}{sampling_flag}"


def save_checkpoint(system, run_name, config, loss_history=None, path="./FullOrderReconstruct"):
    """Save model checkpoint with configuration and loss history"""
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        'model_state_dict': system.model.state_dict(),
        'optimizer_state_dict': system.optim.state_dict(),
        'omega_mean': system.omega_mean,
        'omega_std': system.omega_std,
        'qv_mean': system.qv_mean,
        'qv_std': system.qv_std,
        'coord_min': system.coord_min,
        'coord_max': system.coord_max,
        'output_mean': system.output_mean,
        'output_std': system.output_std,
        'epoch': system.epoch,
        'pretrain_epoch': system.pretrain_epoch,
        'loss_history': loss_history if loss_history else {}
    }

    checkpoint_path = os.path.join(path, f"{run_name}_checkpoint.pth")
    config_path = os.path.join(path, f"{run_name}_config.json")

    torch.save(checkpoint, checkpoint_path)
    config.save(config_path)

    print(f"[INFO] ✅ Checkpoint saved to {checkpoint_path}")
    print(f"[INFO] ✅ Config saved to {config_path}")

    return checkpoint_path, config_path


def load_checkpoint(run_name, path="./FullOrderReconstruct"):
    """Load model checkpoint and configuration"""
    checkpoint_path = os.path.join(path, f"{run_name}_checkpoint.pth")
    config_path = os.path.join(path, f"{run_name}_config.json")

    print(f"[INFO] 🔄 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load configuration
    config = Config.load(config_path)

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Create system
    system = FullOrderPINN(config)

    # Load model parameters
    system.model.load_state_dict(checkpoint['model_state_dict'])
    system.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load normalization stats
    system.omega_mean = checkpoint['omega_mean']
    system.omega_std = checkpoint['omega_std']
    system.qv_mean = checkpoint['qv_mean']
    system.qv_std = checkpoint['qv_std']
    system.coord_min = checkpoint['coord_min']
    system.coord_max = checkpoint['coord_max']
    system.output_mean = checkpoint['output_mean']
    system.output_std = checkpoint['output_std']
    system.epoch = checkpoint['epoch']
    system.pretrain_epoch = checkpoint['pretrain_epoch']

    print("[INFO] ✅ Model and stats restored successfully from checkpoint.")
    return system, config, checkpoint.get('loss_history', {})


def save_loss_history(loss_history, run_name, path="./FullOrderReconstruct"):
    """Save loss history to CSV file"""
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, f"{run_name}_loss_history.csv")

    df = pd.DataFrame(loss_history)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] ✅ Loss history saved to {csv_path}")
    return csv_path


def save_loss_curves(loss_history, run_name, path="./FullOrderReconstruct"):
    """Save loss curves as plots"""
    os.makedirs(path, exist_ok=True)

    epochs = len(loss_history['train_loss'])

    # Main loss plot
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(loss_history['train_loss'], label="Train Loss", alpha=0.7)
    plt.plot(loss_history['test_loss'], label="Test Loss", alpha=0.7)
    if 'physics_loss' in loss_history and any(l > 0 for l in loss_history['physics_loss']):
        plt.plot(loss_history['physics_loss'], label="Physics Loss", alpha=0.7)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Training Loss Curves - {run_name}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Learning rate plot if available
    if 'learning_rate' in loss_history:
        plt.subplot(2, 1, 2)
        plt.plot(loss_history['learning_rate'])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.yscale('log')

    plt.tight_layout()
    plot_path = os.path.join(path, f"{run_name}_loss_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] ✅ Loss curves saved to {plot_path}")
    return plot_path


# ======== Training Function ========
def train_PINN(system, config, run_name):
    """
    Train the FullOrderPINN system with configuration-based parameters

    Args:
        system (FullOrderPINN): The system to train
        config (Config): Training configuration
        run_name (str): Unique identifier for this run

    Returns:
        dict: Loss history containing train_loss, test_loss, physics_loss, learning_rate
    """
    # Initialize loss history
    loss_history = {
        'train_loss': [],
        'test_loss': [],
        'physics_loss': [],
        'learning_rate': []
    }

    # Setup scheduler
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            system.optim,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            verbose=True
        )

    # Setup interrupt handler
    def signal_handler(sig, frame):
        print(f"\n[INFO] ⚠️ Training interrupted at epoch {system.epoch}")
        print("[INFO] 💾 Saving current state...")
        save_checkpoint(system, run_name + "_INTERRUPTED", config, loss_history)
        save_loss_history(loss_history, run_name + "_INTERRUPTED")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Pretraining stage (no physics loss)
    if config.pretrain_epochs:
        print(f"[INFO] Starting pretraining for {config.pretrain_epochs} epochs...")
        system.use_physics_loss = False

        for ep in range(1, config.pretrain_epochs + 1):
            system.pretrain_epoch += 1
            train_loss, test_loss, physics_loss = system.step(config.coord_batch_size)

            current_lr = system.optim.param_groups[0]['lr']

            loss_history['train_loss'].append(train_loss)
            loss_history['test_loss'].append(test_loss)
            loss_history['physics_loss'].append(physics_loss)
            loss_history['learning_rate'].append(current_lr)

            print(f"[PretrainEpoch {ep:04d}] Train Loss = {train_loss:.6f} | "
                  f"Test Loss = {test_loss:.6f} | LR = {current_lr:.2e}")

    # Enable physics loss if originally enabled
    system.use_physics_loss = system.use_physics_loss_original

    # Main training loop
    print(f"[INFO] Starting main training for {config.epochs} epochs...")
    for ep in range(1, config.epochs + 1):
        system.epoch += 1

        if system.use_physics_batch and system.use_physics_loss:
            train_loss, test_loss, physics_loss = system.step_with_physics_batches(config.coord_batch_size)
        else:
            train_loss, test_loss, physics_loss = system.step(config.coord_batch_size)

        current_lr = system.optim.param_groups[0]['lr']

        loss_history['train_loss'].append(train_loss)
        loss_history['test_loss'].append(test_loss)
        loss_history['physics_loss'].append(physics_loss)
        loss_history['learning_rate'].append(current_lr)

        # Print progress
        sampling_info = f" | Sampled {system.sample_size}/{system.total_points} points" if system.use_point_sampling else ""

        if system.use_physics_batch:
            print(f"[Epoch {ep:04d}] Train Loss = {train_loss:.6f} | "
                  f"Test Loss = {test_loss:.6f} | Physics Loss = {physics_loss:.6f} | "
                  f"LR = {current_lr:.2e}{sampling_info}")
        else:
            print(f"[Epoch {ep:04d}] Train Loss = {train_loss:.6f} | "
                  f"Test Loss = {test_loss:.6f} | Physics Loss = {physics_loss:.6f} | "
                  f"LR = {current_lr:.2e}{sampling_info}")

        # Update scheduler
        if scheduler:
            scheduler.step(test_loss)

        # Save checkpoint periodically
        # save_freq = 1000
        # if config.use_physics_loss and config.use_physics_batch:
        #     save_freq = 100
        # if ep % save_freq == 0:
        #     save_checkpoint(system, run_name + f"_epoch{ep}", config, loss_history)
        #     save_loss_history(loss_history, run_name + f"_epoch{ep}")

    return loss_history


# ---------- main entry ----------
if __name__ == "__main__":
    # ======== Configuration ========
    config = Config({
        "SEED": 1024,
        "net_type": "MLP",  # MLP CNN FNO CNO CFNO Transformer
        "model_mode": "deeponet",  # New ModifiedDeepONet
        "hidden_dim": 64,
        "hidden_layers": 5,
        "hidden_branch_dim": 64,
        "hidden_branch_layers": 5,
        "learning_rate": 1e-2,
        "coord_batch_size": 128,
        "use_physics_loss": True,
        "use_physics_batch": True,
        "epochs": 1000,
        "pretrain_epochs": 50,
        "use_scheduler": True,
        "scheduler_patience": 8,
        "scheduler_factor": 0.85,
        "scheduler_min_lr": 1e-6,
        "scaling_factor": 1e-8,
        "use_point_sampling": True,
        "point_sampling_ratio": 0.3,
        "fno_configs": {
            "encoder_type": 'MLP',
            "width": 16,
            "depth": 4,
            "modes": 32,
            "cheb_modes": (16, 16),
            "alpha_init": 0.5,
            "pseudo_grid_size": 64
        }
    })

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Generate unique run name
    run_name = generate_run_name(config)
    print(f"[INFO] Starting run: {run_name}")

    # Create output directory
    os.makedirs("./FullOrderReconstruct", exist_ok=True)

    # Save initial configuration
    config.save(f"./FullOrderReconstruct/{run_name}_config.json")

    # ===== Initialize Model =====
    pinn = FullOrderPINN(config)

    # ===== Train Model =====
    loss_history = train_PINN(pinn, config, run_name)

    # ===== Save Final Results =====
    save_checkpoint(pinn, run_name, config, loss_history, path="./FullOrderReconstruct")
    save_loss_history(loss_history, run_name, path="./FullOrderReconstruct")
    save_loss_curves(loss_history, run_name, path="./FullOrderReconstruct")

    print(f"[INFO] ✅ Training completed successfully! Run name: {run_name}")

    # ===== Test Loaded Model =====
    try:
        loaded_pinn, loaded_config, loaded_history = load_checkpoint(run_name, path="./FullOrderReconstruct/")
        example_coords = torch.tensor([[0.146384, 0.044788, -0.098216]], dtype=torch.float32)
        condition = torch.tensor([630, 0.65])
        result = loaded_pinn(example_coords, condition)
        print("\n[Prediction] Loaded model prediction:", result)

    except FileNotFoundError:
        print("[WARNING] ⚠️ Checkpoint not found for loading test.")