import torch
import pandas as pd
from torch import Tensor
import time


class SmoothKNNInterpolator:
    """
    Optimized SmoothKNNInterpolator:
      - support: modes + mean combined input
      - support: knn index cache to accelerate the later usage
    """
    def __init__(self, coord_path: str, mode_path: str, k: int = 20, modes: int = 8, cache_knn: bool = True):
        # ===== 1. Load coordinates and mode data =====
        coords = pd.read_csv(coord_path, header=None).values  # shape: (N, 3)
        modes = pd.read_csv(mode_path).iloc[:, :modes].values  # shape: (N, modes)

        self.X = torch.tensor(coords, dtype=torch.float32)  # (N, 3)
        self.Y = torch.tensor(modes, dtype=torch.float32)   # (N, modes)
        self.k = k
        self.N, self.D = self.X.shape
        self.cache_knn = cache_knn

        # ===== 2. Optionally cache the KNN indices =====
        self.knn_idx_cache = None
        self.knn_dists_cache = None
        if cache_knn:
            start = time.time()
            print(f"[INFO] 🧮 Precomputing KNN for {self.N} reference points (k={k}) ...")
            with torch.no_grad():
                dists = torch.cdist(self.X, self.X)  # (N, N)
                knn_dists, knn_idx = torch.topk(dists, self.k, dim=1, largest=False)
                self.knn_dists_cache = knn_dists  # (N, k)
                self.knn_idx_cache = knn_idx      # (N, k)
            print(f"[INFO] ✅ KNN cache built in {time.time() - start:.2f} s")

    @staticmethod
    def _smooth_weights(dists: Tensor) -> Tensor:
        eps = 1e-8
        weights = torch.exp(- (dists / (dists[:, -1:] + eps)) ** 2)
        return weights / (weights.sum(dim=1, keepdim=True) + eps)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Interpolate values at input locations x (B, 3)
        Returns interpolated field values (B, modes)
        """
        X, Y = self.X.to(x.device), self.Y.to(x.device)
        x = x.to(X.device)

        # ===== 1. If query points coincide with training points (cached) =====
        if self.cache_knn and x.shape[0] == self.N and torch.allclose(x, X, atol=1e-8):
            knn_dists = self.knn_dists_cache.to(x.device)
            knn_idx = self.knn_idx_cache.to(x.device)
        else:
            x_expanded = x.unsqueeze(1)  # (B, 1, 3)
            dists = torch.norm(X.unsqueeze(0) - x_expanded, dim=2)  # (B, N)
            knn_dists, knn_idx = torch.topk(dists, self.k, dim=1, largest=False)

        knn_values = Y[knn_idx]  # (B, k, modes)
        weights = self._smooth_weights(knn_dists)  # (B, k)
        interpolated = torch.sum(weights.unsqueeze(-1) * knn_values, dim=1)
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
