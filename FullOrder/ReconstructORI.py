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


class PINNSystem:
    def __init__(self, hidden_dim=64, hidden_layers=2, lr=1e-3, use_physics_loss=True, use_physics_batch=False, modes=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self.use_physics_loss_original = use_physics_loss
        self.use_physics_batch = use_physics_batch
        self.use_physics_loss = use_physics_loss
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epoch = 0
        self.pretrain_epoch = 0
        self.modes = modes
        # ===== 1. Load input and normalize =====
        data = pd.read_csv('./EXP.csv')
        omega = torch.tensor(data['omega'].values, dtype=torch.float32)
        qv = torch.tensor(data['qv'].values, dtype=torch.float32)
        split_flag = torch.tensor(data.iloc[:, 2].values, dtype=torch.bool)

        # Save normalization stats
        self.omega_mean, self.omega_std = omega.mean(), omega.std()
        self.qv_mean, self.qv_std = qv.mean(), qv.std()

        # Normalize inputs
        omega_norm = (omega - self.omega_mean) / self.omega_std
        qv_norm = (qv - self.qv_mean) / self.qv_std
        self.conditions = torch.stack([omega_norm, qv_norm], dim=1).to(self.device)

        # Train/Test masks
        self.train_mask = (~split_flag).to(self.device)
        self.test_mask = split_flag.to(self.device)
        self.physics_mask_ = torch.ones_like(split_flag)

        # ===== 2. Load outputs & normalize =====
        self.coeffs_target = {}
        self.coeffs_stats = {}
        self.mean_target = {}
        self.mean_stats = {}

        for name in ['P', 'Ut', 'Ur', 'Uz']:
            coeff = pd.read_csv(f"./ReducedResults/coefficients_{name}.csv").iloc[:, :self.modes].values
            mean = pd.read_csv(f"./ReducedResults/mean_{name}.csv").values.squeeze()

            coeff_torch = torch.tensor(coeff, dtype=torch.float32)
            mean_torch = torch.tensor(mean, dtype=torch.float32)

            coeff_mean = coeff_torch.mean(dim=0)
            coeff_std = coeff_torch.std(dim=0)
            self.coeffs_stats[name] = (coeff_mean.to(self.device), coeff_std.to(self.device))

            mean_mean = mean_torch.mean()
            mean_std = mean_torch.std()
            self.mean_stats[name] = (mean_mean.to(self.device), mean_std.to(self.device))

            self.coeffs_target[name] = ((coeff_torch - coeff_mean) / coeff_std).to(self.device)
            self.mean_target[name] = ((mean_torch - mean_mean) / mean_std).to(self.device)

        # load_ points
        self.coords_all = torch.tensor(pd.read_csv("./coordinate.csv").values, dtype=torch.float32).to(self.device)
        # Material constants
        self.rho = 10278   # kg/m3
        self.mu = 0.0016  # Pa·s
        self.g = 9.8
        # B: training data used in datasets when calculating physics loss
        self.B = 10

        # ===== 3. Networks =====
        def make_net():
            layers = [nn.Linear(2, self.hidden_dim), nn.Tanh()]
            for _ in range(self.hidden_layers - 1):
                layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()]
            layers += [nn.Linear(self.hidden_dim, self.modes+1)]
            return nn.Sequential(*layers).to(self.device)

        self.net_P = make_net()
        self.net_Ut = make_net()
        self.net_Ur = make_net()
        self.net_Uz = make_net()

        # ===== 4. Interpolators =====
        self.interp = {
            name: SmoothKNNInterpolator(
                coord_path='./coordinate.csv',
                mode_path=f'./ReducedResults/modes_{name}.csv',
                modes=self.modes,
                k=3,
            ) for name in ['P', 'Ut', 'Ur', 'Uz']
        }

        # ===== 5. Optimizer =====
        self.optim = torch.optim.Adam(
            list(self.net_P.parameters()) +
            list(self.net_Ut.parameters()) +
            list(self.net_Ur.parameters()) +
            list(self.net_Uz.parameters()),
            lr=self.lr
        )

    # use the model
    def __call__(self, coords, condition, enable_grad=False):
        """
        coords: Tensor of shape (M, 3) for spatial coordinates
        condition: Tensor of shape (2,) [omega, qv] (non-normalized)
        enable_grad: If True, enables gradient tracking (used for physics residuals)

        Returns:
            Dict[str, Tensor]: Dictionary of reconstructed physical fields at coords.
                               Each value is a tensor of shape (M,)
        """
        # Set networks to eval mode
        self.net_P.eval()
        self.net_Ut.eval()
        self.net_Ur.eval()
        self.net_Uz.eval()

        # Normalize condition
        omega, qv = condition
        omega = (omega - self.omega_mean) / self.omega_std
        qv = (qv - self.qv_mean) / self.qv_std
        cond = torch.tensor([[omega, qv]], dtype=torch.float32, requires_grad=enable_grad).to(self.device)

        coords = coords.to(self.device)
        if enable_grad:
            coords.requires_grad_(True)

        # If we don't have to use grad then do not grad it
        fields = {}
        context = torch.enable_grad if enable_grad else torch.no_grad
        with context():
            for name, net in zip(['P', 'Ut', 'Ur', 'Uz'],
                                 [self.net_P, self.net_Ut, self.net_Ur, self.net_Uz]):
                pred = net(cond)  # (1, self.modes+1)
                coeffs_norm = pred[:, :self.modes]  # (1, self.modes)
                mean_norm = pred[:, self.modes]  # (1,)

                coeff_mean, coeff_std = self.coeffs_stats[name]
                mean_mean, mean_std = self.mean_stats[name]

                coeffs = coeffs_norm * coeff_std + coeff_mean  # (1, self.modes)
                mean = mean_norm * mean_std + mean_mean  # (1,)

                mode_values = self.interp[name](coords)  # (M, self.modes)
                field = torch.matmul(mode_values, coeffs.T).squeeze(-1) + mean.item()
                fields[name] = field

        return fields

    def compute_data_loss(self):
        loss_fn = nn.MSELoss()
        train_losses = []
        test_losses = []

        for name, net in zip(['P', 'Ut', 'Ur', 'Uz'],
                             [self.net_P, self.net_Ut, self.net_Ur, self.net_Uz]):
            pred = net(self.conditions)         # (N, self.modes+1)
            coeff_pred = pred[:, :self.modes]
            mean_pred = pred[:, self.modes]

            coeff_true = self.coeffs_target[name]
            mean_true = self.mean_target[name]

            train_mask = self.train_mask
            test_mask = self.test_mask

            train_losses.append(loss_fn(coeff_pred[train_mask], coeff_true[train_mask]))
            train_losses.append(loss_fn(mean_pred[train_mask], mean_true[train_mask]))

            test_losses.append(loss_fn(coeff_pred[test_mask], coeff_true[test_mask]))
            test_losses.append(loss_fn(mean_pred[test_mask], mean_true[test_mask]))

        return sum(train_losses), sum(test_losses)

    def compute_physics_residuals(self, coords, condition):
        """
        coords: Tensor (N, 3) - [r, theta, z]
        condition: Tensor (1, 2) - [omega, qv] (normalized)
        Returns: physics residual loss
        """
        coords = coords.detach().clone().requires_grad_()
        r = coords[:, 0:1]

        omega, qv = condition[0, 0], condition[0, 1]  # normalized omega, qv

        # Unnormalize for real-world omega (from r/min to rad/s)
        omega_real = -(omega * self.omega_std + self.omega_mean) * 2 * torch.pi / 60  # rad/s

        # Predict fields at coords under condition
        fields = self.__call__(coords, condition[0], enable_grad=True)  # [P, Ut, Ur, Uz]
        p = fields['P'].unsqueeze(1)
        ut = fields['Ut'].unsqueeze(1) - omega*r    # transfer into the relative velocity
        ur = fields['Ur'].unsqueeze(1)
        uz = fields['Uz'].unsqueeze(1)

        def grad(outputs, inputs):
            return autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                                 create_graph=True, retain_graph=True)[0]

        # Derivatives
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

        mse = nn.MSELoss()

        physics_loss = mse(FC, torch.zeros_like(FC)) + \
                       mse(FR, torch.zeros_like(FR)) + \
                       mse(FTH, torch.zeros_like(FTH)) + \
                       mse(FZ, torch.zeros_like(FZ))
        return physics_loss / 1e6

    def compute_hidden_constraint_loss(self, coord_batch_size=64):
        if not self.use_physics_loss:
            return torch.tensor(0.0, dtype=torch.float32)
        idx = torch.nonzero(self.train_mask).squeeze()
        sample_idx = idx[torch.randperm(len(idx))[:self.B]]
        # random sampling to improve efficiency
        coords_all = self.coords_all[torch.randperm(self.coords_all.shape[0])[:coord_batch_size]]
        physics_loss = 0.0
        for i in sample_idx:
            cond = self.conditions[i].unsqueeze(0)  # (1, 2)
            loss = self.compute_physics_residuals(coords_all.clone(), cond)
            physics_loss += loss
        return physics_loss / self.B

    # compute loss with batch
    def compute_hidden_constraint_loss_batch(self, coords_batch, condition_batch):
        if not self.use_physics_loss:
            return torch.tensor(0.0, dtype=torch.float32)
        physics_loss = 0.0
        for i in range(condition_batch.shape[0]):
            cond = condition_batch[i:i+1]
            loss = self.compute_physics_residuals(coords_batch.clone(), cond)
            physics_loss += loss
        return physics_loss / condition_batch.shape[0]

    def compute_total_loss(self, coord_batch_size=64):
        train_loss, test_loss = self.compute_data_loss()
        physics_loss = self.compute_hidden_constraint_loss(coord_batch_size=coord_batch_size)
        total_loss = train_loss + physics_loss
        return total_loss, train_loss.item(), test_loss.item(), physics_loss.item()

    def step_with_physics_batches(self, coord_batch_size=64):
        train_loss_total = 0.0
        test_loss_total = 0.0
        physics_loss_total = 0.0

        idx = torch.nonzero(self.train_mask).squeeze()
        sample_idx = idx[torch.randperm(len(idx))[:self.B]]
        condition_batch = self.conditions[sample_idx]  # (B, 2)
        coords_all = self.coords_all[torch.randperm(self.coords_all.shape[0])]
        num_coords = coords_all.shape[0]
        num_batches = (num_coords + coord_batch_size - 1) // coord_batch_size


        for i in range(num_batches):
            start = i * coord_batch_size
            end = min((i + 1) * coord_batch_size, num_coords)
            coords_batch = coords_all[start:end]
            self.optim.zero_grad()
            train_loss, test_loss = self.compute_data_loss()
            physics_loss = self.compute_hidden_constraint_loss_batch(coords_batch, condition_batch)
            total_loss = train_loss + physics_loss
            total_loss.backward()
            self.optim.step()

            train_loss_total += train_loss.item()
            test_loss_total += test_loss.item()
            physics_loss_total += physics_loss.item()

            print(f"    [Batch {i + 1}/{num_batches}] Train Loss: {train_loss.item():.6f}, "
                  f"Test Loss: {test_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}")

        return train_loss_total / num_batches, test_loss_total / num_batches

    def step(self, coord_batch_size=64):
        self.optim.zero_grad()
        total_loss, train_loss, test_loss, phy_loss = self.compute_total_loss(coord_batch_size=coord_batch_size)
        total_loss.backward()
        self.optim.step()
        return train_loss, test_loss, phy_loss


# ======== Seed Function ========
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# save model
def save_checkpoint(system: PINNSystem, seed: int, path: str = "ReconstructORI"):
    checkpoint = {
        'net_P_state_dict': system.net_P.state_dict(),
        'net_Ut_state_dict': system.net_Ut.state_dict(),
        'net_Ur_state_dict': system.net_Ur.state_dict(),
        'net_Uz_state_dict': system.net_Uz.state_dict(),
        'optimizer_state_dict': system.optim.state_dict(),
        'omega_mean': system.omega_mean,
        'omega_std': system.omega_std,
        'qv_mean': system.qv_mean,
        'qv_std': system.qv_std,
        'coeffs_stats': system.coeffs_stats,
        'mean_stats': system.mean_stats,
    }
    filename = path+f"/Checkpoint_SEED{seed}_LR{system.lr}_HD{system.hidden_dim}_HL{system.hidden_layers}_Epoch{system.epoch}_WithPhysics{int(system.use_physics_loss)}_WithBatch{(int(system.use_physics_batch))}"
    if system.pretrain_epoch:
        filename += f"_Pretrain{system.pretrain_epoch}"
    filename += ".pth"
    torch.save(checkpoint, filename)
    print(f"[INFO] ✅ Model checkpoint saved to {filename}")


# Read Model
def load_checkpoint(path: str):
    print(f"[INFO] 🔄 Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # ===== Parse the Filenames =====
    filename = os.path.basename(path)
    pattern = r"SEED(?P<seed>\d+)_LR(?P<lr>[\d\.eE+-]+)_HD(?P<hd>\d+)_HL(?P<hl>\d+)_Epoch(?P<epoch>\d+)_WithPhysics(?P<phys>\d+)_WithBatch(?P<batch>\d+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"[ERROR] ❌ Cannot parse hyperparameters from filename: {filename}")
    parsed = match.groupdict()
    seed = int(parsed['seed'])
    lr = float(parsed['lr'])
    hidden_dim = int(parsed['hd'])
    hidden_layers = int(parsed['hl'])
    use_physics_loss = bool(int(parsed['phys']))
    use_physics_batch = bool(int(parsed['batch']))

    # =====Set Seed=====
    set_seed(seed)

    # ===== Create PINN System =====
    system = PINNSystem(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        lr=lr,
        use_physics_loss=use_physics_loss,
        use_physics_batch=use_physics_batch,
        modes=len(checkpoint['coeffs_stats']['P'][0]),
    )

    # ===== LoadParas =====
    system.net_P.load_state_dict(checkpoint['net_P_state_dict'])
    system.net_Ut.load_state_dict(checkpoint['net_Ut_state_dict'])
    system.net_Ur.load_state_dict(checkpoint['net_Ur_state_dict'])
    system.net_Uz.load_state_dict(checkpoint['net_Uz_state_dict'])

    system.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    system.omega_mean = checkpoint['omega_mean']
    system.omega_std = checkpoint['omega_std']
    system.qv_mean = checkpoint['qv_mean']
    system.qv_std = checkpoint['qv_std']
    system.coeffs_stats = checkpoint['coeffs_stats']
    system.mean_stats = checkpoint['mean_stats']
    system.epoch = int(parsed['epoch'])

    print("[INFO] ✅ Model and stats restored successfully from checkpoint.")
    return system


def save_loss_curve(train_loss, test_loss, SEED, learning_rate, hidden_dim,
                    hidden_layers, use_physics_loss, use_physics_batch, pretrain,
                    save_dir="ReconstructORI"):
    epochs = len(train_loss)
    os.makedirs(save_dir, exist_ok=True)

    filename = (f"Losscurve_SEED{SEED}_LR{learning_rate}_HD{hidden_dim}_"
                f"HL{hidden_layers}_Epoch{epochs}_WithPhysics{int(use_physics_loss)}"
                f"_WithBatch{int(use_physics_batch)}{'' if pretrain else '_Pretrain'+str(pretrain)}.png")
    filepath = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss", marker='o', markersize=3)
    plt.plot(test_loss, label="Test Loss", marker='s', markersize=3)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Test Loss Curve")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"[INFO] Loss curve saved to: {filepath}")
    return filepath


# ======== Train ========
def train_PINN(system: PINNSystem, epochs: int = 9999, coord_batch_size: int = 64, pretrain_epoch: int = 0,
               use_scheduler: bool = True, scheduler_patience: int = 50, scheduler_factor: float = 0.75,
               scheduler_min_lr: float = 1e-6):
    """
    Train the PINN system with optional pretraining and learning rate scheduler.

    Args:
        system (PINNSystem): The system to train.
        epochs (int): Number of main training epochs.
        coord_batch_size (int): Batch size for coordinate sampling.
        pretrain_epoch (int): Number of pretraining epochs without physics loss.
        use_scheduler (bool): Whether to enable LR scheduler.
        scheduler_patience (int): Patience before reducing LR.
        scheduler_factor (float): LR decay factor.
        scheduler_min_lr (float): Minimum learning rate.

    Returns:
        train_loss_history (list): Training loss per epoch.
        test_loss_history (list): Test loss per epoch.
    """
    train_loss_history = []
    test_loss_history = []

    # Scheduler setup
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            system.optim,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
            verbose=True
        )
    # Pretraining stage (no physics loss)
    if pretrain_epoch:
        print("[INFO] Starting pretraining...")
        system.use_physics_loss = False
        for ep in range(1, pretrain_epoch + 1):
            system.pretrain_epoch += 1
            train_loss, test_loss, _ = system.step(coord_batch_size)
            print(f"[PretrainEpoch {ep:04d}] Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f}")
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
    # Enable physics loss if originally enabled
    system.use_physics_loss = system.use_physics_loss_original
    # Main training loop
    print("[INFO] Starting main training...")
    for ep in range(1, epochs + 1):
        system.epoch += 1
        if system.use_physics_batch:
            train_loss, test_loss = system.step_with_physics_batches(coord_batch_size)
            print(f"[Epoch {ep:04d}] Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f}")
            # This Save
            save_checkpoint(system, seed=SEED, path='./ReconstructORI/Step')
            save_loss_curve(train_loss_history, test_loss_history,
                            SEED, system.lr, system.hidden_dim,
                            system.hidden_layers, system.use_physics_loss,
                            system.use_physics_batch, system.pretrain_epoch,
                            save_dir="./ReconstructORI/Step"
                            )
        else:
            train_loss, test_loss, phy_loss = system.step(coord_batch_size)
            print(f"[Epoch {ep:04d}] Train Loss = {train_loss:.6f} | Test Loss = {test_loss:.6f} "
                  f"| Physics Loss = {phy_loss:.6f}")
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        # Scheduler step based on test loss
        if scheduler:
            scheduler.step(test_loss)
    return train_loss_history, test_loss_history


if __name__ == "__main__":
    # ======== Main Entry, Train Your Model Here ========
    MODES = 4
    SEED = 42
    set_seed(SEED)
    # Hyper Parameters
    hidden_dim = 30
    hidden_layers = 2
    learning_rate = 1e-2
    coord_batch_size = 128
    use_physics_loss = True
    use_physics_batch = True    # slow but accurate, can decrease the epochs
    epochs = 63
    pretrain_epochs = 50

    # ===== Initialize Model=====
    pinn = PINNSystem(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        use_physics_loss=use_physics_loss,
        lr=learning_rate,
        use_physics_batch=use_physics_batch,
        modes=MODES,
    )

    # ===== Load Checkpoint if Exists =====
    checkpoint_path = "ReconstructORI"
    # ===== Train =====
    train_loss, test_loss = train_PINN(
        pinn, epochs=epochs, coord_batch_size=coord_batch_size, pretrain_epoch=pretrain_epochs,
        use_scheduler=True
    )
    # ===== Save New Checkpoint =====
    save_checkpoint(pinn, SEED, checkpoint_path)
    save_loss_curve(
        train_loss=train_loss,
        test_loss=test_loss,
        SEED=SEED,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        use_physics_loss=use_physics_loss,
        use_physics_batch=use_physics_batch,
        pretrain=pretrain_epochs
    )
    try:
        if not pretrain_epochs:
            pinn = load_checkpoint(checkpoint_path+f"/Checkpoint_SEED{SEED}_LR{learning_rate}_HD{hidden_dim}_HL{hidden_layers}_MO{MODES}_Epoch{epochs}_WithPhysics{int(use_physics_loss)}_WithBatch{(int(use_physics_batch))}.pth")
        else:
            pinn = load_checkpoint(checkpoint_path+f"/Checkpoint_SEED{SEED}_LR{learning_rate}_HD{hidden_dim}_HL{hidden_layers}_MO{MODES}_Epoch{epochs}_WithPhysics{int(use_physics_loss)}_WithBatch{(int(use_physics_batch))}_Pretrain{pretrain_epochs}.pth")
        example_coords = torch.tensor([[0.146384, 0.044788, -0.098216]], dtype=torch.float32)
        condition = torch.tensor([630, 0.65])
        result = pinn(example_coords, condition)
        print("\n[Prediction] Loaded model prediction:", result)
    except FileNotFoundError:
        print("[WARNING] ⚠️ No checkpoint found. Training from scratch.")
