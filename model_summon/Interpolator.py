import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1.Seed is the only way to reproduce the model!
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Normalizer class for scaling modal outputs
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0] = 1  # To avoid division by zero

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# 2. Define the Network Structure
class ModalNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, dropout=0.2, lr=1e-3, hidden_dims=[64, 64], normalizer=None, device='cuda'):
        super(ModalNet, self).__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            # Preparing for dealing with large scale simulation data
            layers.append(nn.Dropout(dropout))  # Add some regularization,
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, output_dim))  # 4 outputs
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.normalizer = normalizer  # Save Normalizer instance for inverse transform
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

    # a step update
    def train_step(self, x_batch, y_batch):
        self.train()
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        self.optimizer.zero_grad()
        preds = self.forward(x_batch)
        loss = self.loss_fn(preds, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # predict with model
    def predict(self, x, denormalize=True):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            preds = self.forward(x)
            if denormalize and self.normalizer is not None:
                preds_np = preds.cpu().numpy()
                preds_np = self.normalizer.inverse_transform(preds_np)
                preds = torch.tensor(preds_np, dtype=preds.dtype)
            return preds


def load_data(coord_file, mode_file):
    # Load coordinates, no header
    coords = pd.read_csv(coord_file, header=None).values.astype(np.float32)
    # Load modal data, with header, only first 4 columns needed
    modes = pd.read_csv(mode_file).iloc[:, :4].values.astype(np.float32)
    return coords, modes


# check the trend after training
def evaluate_metrics(model, dataloader, normalizer, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            preds = model.predict(xb, denormalize=True).cpu().numpy()
            targets = normalizer.inverse_transform(yb.cpu().numpy())
            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    corr_list = []
    mae_list = []
    rmse_list = []
    re_list = []

    for i in range(preds.shape[1]):
        corr, _ = pearsonr(preds[:, i], targets[:, i])
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        re = np.mean(np.abs((targets[:, i]-preds[:, i])/(preds[:, i]+1e-9)))
        corr_list.append(corr)
        mae_list.append(mae)
        rmse_list.append(rmse)
        re_list.append(re)
    return corr_list, mae_list, rmse_list, re_list


# model training (Switch the data_name)
def train_model(data_name='Ur', epochs=2000, batch_size=128, dropout=0, test_ratio=0.2, lr=1e-2, hidden_dims=[20, 20, 20, 20], seed=25):
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(seed)
    coord_file = './coordinate.csv'
    mode_file = f'./ReducedResults/modes_{data_name}.csv'
    model_save_path = f'./Interpolator/Modal_{data_name}.pth'

    # Load data
    X, Y = load_data(coord_file, mode_file)

    # Normalize inputs
    input_normalizer = Normalizer()
    input_normalizer.fit(X)
    X_norm = input_normalizer.transform(X)

    # Initialize Normalizer and normalize outputs
    normalizer = Normalizer()
    normalizer.fit(Y)
    Y_norm = normalizer.transform(Y)

    dataset = TensorDataset(torch.tensor(X_norm), torch.tensor(Y_norm))
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model init with normalizer
    model = ModalNet(input_dim=3, output_dim=4, dropout=dropout, lr=lr, normalizer=normalizer, hidden_dims=hidden_dims, device=device)
    # decay of the lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', factor=0.6, patience=30, verbose=True
    )

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            loss = model.train_step(xb, yb)
            batch_losses.append(loss)
        avg_train_loss = np.mean(batch_losses)

        model.eval()
        test_batch_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model.predict(xb, denormalize=False)
                loss = model.loss_fn(preds.to(device), yb).item()
                test_batch_losses.append(loss)
        avg_test_loss = np.mean(test_batch_losses)
        # Update scheduler
        scheduler.step(avg_test_loss)
        # record loss
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        if avg_train_loss < 1e-6:
            break

        # if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")

    # Save model and normalizer stats
    os.makedirs('./Interpolator', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    np.savez(f'./Interpolator/InputNormalizer_{data_name}.npz', mean=input_normalizer.mean, std=input_normalizer.std)
    np.savez(f'./Interpolator/Normalizer_{data_name}.npz', mean=normalizer.mean, std=normalizer.std)
    print(f"Model saved to {model_save_path}")
    print(f"Normalizer stats saved to ./Interpolator/Normalizer_{data_name}.npz")

    corrs, maes, rmses, re = evaluate_metrics(model, test_loader, normalizer, device)
    print("\nEvaluation on Denormalized Predictions:")
    for i in range(len(corrs)):
        print(
            f"Mode {i + 1}: Pearson Corr = {corrs[i]:.6f}, MAE = {maes[i]:.6f}, RMSE = {rmses[i]:.6f}, RE = {re[i]:.6f}")

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(train_losses)+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (normalized)')
    plt.yscale("log")
    plt.title(f'Training and Testing Loss for {data_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Interpolator/LossCurve_{data_name}.png')
    plt.show()


if __name__ == '__main__':
    train_model()
