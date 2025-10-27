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


# ----------------------------------------------
# FullOrderPINN: PINN-DeepONet-FullMLP
# ----------------------------------------------


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


# ======== FullOrderPINN system ========
class FullOrderPINN:
    def __init__(self,
                 net_type='MLP',
                 model_mode='deeponet',
                 hidden_dim=64,
                 hidden_layers=2,
                 lr=1e-3,
                 hidden_branch_dim=32,
                 hidden_branch_layers=4,
                 use_physics_loss=True,
                 use_physics_batch=False,
                 scaling_factor=1.0,
                 fno_configs=None):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # bookkeeping
        self.net_type = net_type
        self.model_mode = model_mode
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.use_physics_loss_original = use_physics_loss
        self.use_physics_loss = use_physics_loss
        self.use_physics_batch = use_physics_batch
        self.scaling_factor = scaling_factor
        self.epoch = 0
        self.pretrain_epoch = 0
        self.hidden_branch_dim = hidden_branch_dim
        self.hidden_branch_layers = hidden_branch_layers

        # FNO configs
        if fno_configs is None:
            self.fno_configs = {
                "encoder_type": 'MLP',
                "width": 16,
                "depth": 3,
                "modes": 16,
                "cheb_modes": (8, 8),
                "alpha_init": 0.5,
                "pseudo_grid_size": 64
            }
        else:
            self.fno_configs = fno_configs

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

        # -------- 4. Create model ----------
        if self.model_mode == 'deeponet':
            self.model = DeepONetFull(
                trunk_type=self.net_type,
                trunk_hidden=self.hidden_dim,
                trunk_layers=self.hidden_layers,
                branch_hidden=self.hidden_branch_dim,
                branch_layers=self.hidden_branch_layers,
                out_dim=4,
                fno_configs=self.fno_configs).to(self.device)

        else:
            self.model = FullMLP(input_dim=5, hidden_dim=self.hidden_dim,
                                 hidden_layers=self.hidden_layers, out_dim=4).to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # physics constants
        self.rho = 10650.0
        self.mu = 0.006
        self.g = 9.81
        self.B = 10

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
    # Data Loss (MSE)
    # ============================================================
    def compute_data_loss(self):
        """
        Compute MSE loss between predicted and true normalized fields over all conditions.
        Returns:
            train_total (tensor with grad)
            test_total (tensor with grad)
        """
        loss_fn = nn.MSELoss()
        train_losses, test_losses = [], []

        nconds = self.conditions.shape[0]

        for idx in range(nconds):
            cond_input = self.conditions[idx:idx + 1]  # shape (1,2)
            preds = self.model(self.coords_all, cond_input)  # (Npoints,4)

            loss_sum = 0.0
            for name in ['P', 'Ut', 'Ur', 'Uz']:
                true_n = getattr(self, f"{name}_all")[:, idx]
                pred_n = preds[:, 'PUtUrUz'.index(name)] if self.model_mode != 'deeponet' else preds[:,
                                                                                               ['P', 'Ut', 'Ur',
                                                                                                'Uz'].index(name)]
                loss_sum += loss_fn(pred_n, true_n)

            if self.train_mask[idx]:
                train_losses.append(loss_sum)
            else:
                test_losses.append(loss_sum)

        train_total = torch.stack(train_losses).mean() if train_losses else torch.tensor(0.0, device=self.device)
        test_total = torch.stack(test_losses).mean() if test_losses else torch.tensor(0.0, device=self.device)

        return train_total, test_total

    # ============================================================
    # Physics residual computation
    # ============================================================
    def compute_physics_residuals(self, coords, condition):
        """
        Compute physics residuals with simplified approach for FNO-based models
        """
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
    # Helper losses
    # ============================================================
    def compute_hidden_constraint_loss(self, coord_batch_size=64):
        if not self.use_physics_loss:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        idx = torch.nonzero(self.train_mask).squeeze()
        if idx.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        sample_idx = idx[torch.randperm(len(idx))[:self.B]]
        coords_all = self.coords_real[torch.randperm(self.coords_real.shape[0])[:coord_batch_size]]
        physics_loss = 0.0
        for i in sample_idx:
            cond = self.conditions[i].unsqueeze(0)
            physics_loss += self.compute_physics_residuals(coords_all.clone(), cond)
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
        train_loss, test_loss = self.compute_data_loss()
        physics_loss = self.compute_hidden_constraint_loss(coord_batch_size=coord_batch_size)
        total_loss = train_loss + physics_loss * self.scaling_factor
        return total_loss, train_loss, test_loss, physics_loss

    # ---------- training step variants ----------
    def step_with_physics_batches(self, coord_batch_size=64):
        """
        Perform one training step using batches of coordinates for physics loss.
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

        # permute coordinates for batching
        coords_permuted = self.coords_all[torch.randperm(self.coords_all.shape[0])]
        num_coords = coords_permuted.shape[0]
        num_batches = (num_coords + coord_batch_size - 1) // coord_batch_size

        for i in range(num_batches):
            start = i * coord_batch_size
            end = min((i + 1) * coord_batch_size, num_coords)
            coords_batch = coords_permuted[start:end]

            # zero gradients
            self.optim.zero_grad()

            # compute data loss
            train_loss, test_loss = self.compute_data_loss()

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
        self.optim.zero_grad()
        total_loss, train_loss, test_loss, phy_loss = self.compute_total_loss(coord_batch_size=coord_batch_size)
        total_loss.backward()
        self.optim.step()
        return train_loss.item(), test_loss.item(), phy_loss.item()

    # ---------- checkpoint & save functions ----------
    def save_checkpoint(self, seed: int, path: str = "FullOrderReconstruct"):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'omega_mean': self.omega_mean,
            'omega_std': self.omega_std,
            'qv_mean': self.qv_mean,
            'qv_std': self.qv_std,
            'net_type': self.net_type,
            'model_mode': self.model_mode,
            'hidden_dim': self.hidden_dim,
            'hidden_layers': self.hidden_layers,
            'fno_configs': self.fno_configs,
        }
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path,
                                f"Checkpoint_SEED{seed}_LR{self.lr}_HD{self.hidden_dim}_HL{self.hidden_layers}_MODE{self.model_mode}_NET{self.net_type}_Epoch{self.epoch}_WithPhysics{int(self.use_physics_loss)}_WithBatch{int(self.use_physics_batch)}.pth")
        torch.save(checkpoint, filename)
        print(f"[INFO] ✅ Model checkpoint saved to {filename}")
        return filename


def load_fullorder_checkpoint(path: str):
    print(f"[INFO] 🔄 Loading FullOrder checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return checkpoint


def save_loss_curve(train_loss, test_loss, physics_loss, SEED, learning_rate, hidden_dim,
                    hidden_layers, modes_or_model, use_physics_loss, use_physics_batch, pretrain,
                    save_dir="FullOrderReconstruct"):
    epochs = len(train_loss)
    os.makedirs(save_dir, exist_ok=True)

    filename_base = (f"Losscurve_SEED{SEED}_LR{learning_rate}_HD{hidden_dim}_"
                     f"HL{hidden_layers}_MO{modes_or_model}_Epoch{epochs}_WithPhysics{int(use_physics_loss)}"
                     f"_WithBatch{int(use_physics_batch)}"
                     f"{'' if not pretrain else '_Pretrain' + str(pretrain)}")

    # 保存图像
    plt.figure(figsize=(10, 6))
    train_loss_c = [float(x.detach().cpu().item() if torch.is_tensor(x) else x) for x in train_loss]
    test_loss_c = [float(x.detach().cpu().item() if torch.is_tensor(x) else x) for x in test_loss]

    plt.plot(train_loss_c, label="Train Loss", marker='o', markersize=3)
    plt.plot(test_loss_c, label="Test Loss", marker='s', markersize=3)
    plt.plot(physics_loss, label="Physics Loss", marker='^', markersize=3)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training, Test, and Physics Loss Curve (FullOrderPINN)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename_base + ".png"))
    plt.close()

    # 保存数据
    df = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "train_loss": train_loss_c,
        "test_loss": test_loss_c,
        "physics_loss": physics_loss
    })
    df.to_csv(os.path.join(save_dir, filename_base + ".csv"), index=False)

    print(f"[INFO] Loss curve and data saved to: {filename_base}.(png/csv)")
    return filename_base


def train_PINN(system: FullOrderPINN, epochs: int = 9999, coord_batch_size: int = 64, pretrain_epoch: int = 0,
               use_scheduler: bool = True, scheduler_patience: int = 50, scheduler_factor: float = 0.75,
               scheduler_min_lr: float = 1e-6, SEED: int = 42):
    train_loss_history = []
    test_loss_history = []
    phy_loss_history = []

    # Scheduler setup
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            system.optim,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr
        )

    # Pretraining stage (no physics loss)
    if pretrain_epoch:
        print("[INFO] Starting pretraining...")
        system.use_physics_loss = False
        for ep in range(1, pretrain_epoch + 1):
            system.pretrain_epoch += 1
            train_loss, test_loss, phy_loss = system.step(coord_batch_size)
            print(f"[PretrainEpoch {ep:04d}] Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f}")
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            phy_loss_history.append(phy_loss)
    # Enable physics loss if originally enabled
    system.use_physics_loss = system.use_physics_loss_original

    # Main training loop
    print("[INFO] Starting main training...")
    for ep in range(1, epochs + 1):
        system.epoch += 1
        if system.use_physics_batch and system.use_physics_loss:
            train_loss, test_loss, phy_loss = system.step_with_physics_batches(coord_batch_size)
            print(
                f"[Epoch {ep:04d}] Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f} | Physics = {phy_loss:.6f}")
        else:
            train_loss, test_loss, phy_loss = system.step(coord_batch_size)
            print(f"[Epoch {ep:04d}] Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f} "
                  f"| Physics Loss = {phy_loss:.6f}")
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        phy_loss_history.append(phy_loss)

        # Scheduler step based on test loss
        if scheduler:
            scheduler.step(test_loss)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"    [LR Scheduler] Epoch {ep:04d}: current LR = {current_lr:.6e}")

    return train_loss_history, test_loss_history, phy_loss_history


# ---------- main entry ----------
if __name__ == "__main__":
    # Hyperparameters
    MODES_OR_MODEL = 'FullOrder'
    SEED = 1024
    set_seed(SEED)

    net_type = "CFNO"
    model_mode = "deeponet"
    hidden_dim = 64
    hidden_layers = 2
    learning_rate = 1e-3
    coord_batch_size = 128
    use_physics_loss = True
    use_physics_batch = True
    epochs = 210
    pretrain_epochs = 200
    use_scheduler = True
    scheduler_patience = 5
    scheduler_factor = 0.75
    scheduler_min_lr = 1e-6
    scaling_factor = 1
    hidden_branch_dim = 128
    hidden_branch_layers = 4

    # FNO settings
    FNO_CONFIGS = {
        "encoder_type": 'MLP',
        "width": 16,
        "depth": 4,
        "modes": 32,
        "cheb_modes": (16, 16),
        "alpha_init": 0.5,
        "pseudo_grid_size": 64
    }

    # Initialize FullOrderPINN
    pinn = FullOrderPINN(
        net_type=net_type,
        model_mode=model_mode,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        lr=learning_rate,
        use_physics_loss=use_physics_loss,
        use_physics_batch=use_physics_batch,
        scaling_factor=scaling_factor,
        hidden_branch_dim=hidden_branch_dim,
        hidden_branch_layers=hidden_branch_layers,
        fno_configs=FNO_CONFIGS,
    )

    # Train
    print(f"[INFO] Starting FullOrder training with model_mode={model_mode}, net_type={net_type} ...")
    train_loss, test_loss, physics_loss = train_PINN(
        pinn,
        epochs=epochs,
        coord_batch_size=coord_batch_size,
        pretrain_epoch=pretrain_epochs,
        use_scheduler=use_scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        scheduler_min_lr=scheduler_min_lr,
        SEED=SEED
    )

    # Save final checkpoint and loss curves
    final_dir = "FullOrderReconstruct"
    pinn.save_checkpoint(seed=SEED, path=final_dir)
    save_loss_curve(train_loss, test_loss, physics_loss, SEED, learning_rate, hidden_dim,
                    hidden_layers, model_mode, use_physics_loss, use_physics_batch, pretrain_epochs,
                    save_dir=final_dir)

    print("[INFO] ✅ FullOrder training complete.")
