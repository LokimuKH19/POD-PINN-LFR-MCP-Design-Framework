import torch
import pandas as pd
from torch import Tensor
import time


class SmoothKNNInterpolator:
    def __init__(self, coord_path: str, mode_path: str, k: int = 20, modes: int = 8):
        # Load coordinates and modes
        coords = pd.read_csv(coord_path, header=None).values  # shape: (N, 3)
        modes = pd.read_csv(mode_path).iloc[:, :modes].values     # shape: (N, 4)

        self.X = torch.tensor(coords, dtype=torch.float32)    # (N, 3)
        self.Y = torch.tensor(modes, dtype=torch.float32)     # (N, 4)
        self.k = k

        self.N, self.D = self.X.shape

    @staticmethod
    def _smooth_weights(dists: Tensor) -> Tensor:
        """
        Use Gaussian weights to assign soft importance.
        Shape: (B, k)
        """
        eps = 1e-8
        weights = torch.exp(- (dists / (dists[:, -1:] + eps)) ** 2)
        return weights / (weights.sum(dim=1, keepdim=True) + eps)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Interpolate values at input locations x (denoted as B).
        x: (B, 3)
        Return: (B, 4)
        """
        X = self.X.to(x.device)
        Y = self.Y.to(x.device)

        x_expanded = x.unsqueeze(1)  # (B, 1, 3)
        dists = torch.norm(X.unsqueeze(0) - x_expanded, dim=2)  # (B, N)

        # Find k nearest neighbors
        knn_dists, knn_idx = torch.topk(dists, self.k, dim=1, largest=False)  # (B, k)
        knn_values = Y[knn_idx]  # (B, k, 4)

        # Compute soft weights
        weights = self._smooth_weights(knn_dists)  # (B, k)

        # Weighted sum
        interpolated = torch.sum(weights.unsqueeze(-1) * knn_values, dim=1)  # (B, 4)
        return interpolated


if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    # Instantiate the interpolator
    start = time.time()
    interp = SmoothKNNInterpolator(
        coord_path='./coordinate.csv',
        mode_path='./ReducedResults/modes_Uz.csv',
        k=3    # Using 3-nearest-neighbor
    )
    print("Interpolator Ready", time.time() - start)

    # Test input
    x = torch.tensor([
        [0.146384, 0.044788, -0.098216],
        [0.146383, 0.032716, -0.098214],
        [0.1463835, 0.038716, -0.098215]
    ], requires_grad=True)

    # Interpolation
    epsilon_shift = 1e-8
    x_perturbed = x + epsilon_shift * torch.randn_like(x)    # to get 2nd derivatives
    y = interp(x_perturbed)
    print("Interpolated Result:", y, time.time() - start)

    # First-order gradient
    grad1 = torch.autograd.grad(y[:, 0].sum(), x, create_graph=True)[0]
    print("1st-order Gradient (∂y/∂x):", grad1)

    # Second-order Hessian (B, 3, 3)
    grads = []
    for i in range(x.shape[1]):
        grad2 = torch.autograd.grad(grad1[:, i].sum(), x, create_graph=True)[0]
        grads.append(grad2.unsqueeze(1))
    hessian = torch.cat(grads, dim=1)
    print("2nd-order Derivatives (Hessian):", hessian)
