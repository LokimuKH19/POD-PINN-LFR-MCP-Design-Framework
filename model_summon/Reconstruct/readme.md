# Result Discussion of POD-PINN with KNN Interpolator

This is a detailed discussion of the enhanced program `../ReconstructORI.py`(The method we adopted in the paper and with better performance, reproductivity and more readable and debugging friendlier code). It leverages pre-computed modal bases for the reconstruction of physical fields, and introduces a carefully balanced combination of **data loss** and **physics loss**.

---

## 🔧 Key Features

- **ROM-Based Interpolation**: Predicts reduced coefficients and reconstructs full fields using interpolated modes (`SmoothKNNInterpolator`).
- **Normalized Input/Output**: All physical quantities are normalized to stabilize training.
- **Physics Loss Supervision**: Uses physics residuals (continuity + momentum) as *regularization*, not strict minimization.
- **Gradient Flow Enabled Call**: Model supports gradient-enabled forward pass via `enable_grad=True` for computing PDE residuals.
- **Coordinate Shift for Stability**: All coordinates are shifted by `1e-6` to avoid gradient instabilities in physics loss.
- **Physics/Data Loss Balancing**: Physics loss is divided by `1e7` to match the scale of data loss and prevent it from dominating.
- **Batch-based Physics Loss**: Efficient sampling strategy over spatial and parametric dimensions to accelerate training.
- **Flexible Training Modes**: Supports both random point sampling and full coordinate-batch training (configurable).

---

## 📦 Data Requirements

The following files are expected in the working directory:

```
./EXP.csv                    # Experimental parameters: omega, qv, and split flags
./coordinate.csv             # All spatial coordinates (r, theta, z)
./ReducedResults/coefficients_{P,Ut,Ur,Uz}.csv  # Modal coefficients for each variable
./ReducedResults/mean_{P,Ut,Ur,Uz}.csv          # Mean value for each variable
./ReducedResults/modes_{P,Ut,Ur,Uz}.csv         # Mode bases (M x 4 matrix per variable)
```

---

## 🧠 Model Structure

Each variable (`P`, `Ut`, `Ur`, `Uz`) is predicted by a shared neural network architecture:

- Input: 2D normalized condition vector `[omega, qv]`
- Output: 4 normalized modal coefficients + 1 normalized mean
- Hidden Size: Configurable (`hidden_dim`, default = 64)
- Activation: `Tanh`

Reconstruction uses:
```python
field = torch.matmul(interpolated_modes, predicted_coeffs.T).squeeze() + mean
```

---

## ⚙️ Training Logic

Two types of loss are computed:

### 1. Data Loss
Supervised regression on reduced coefficients and means.

### 2. Physics Loss
Computed from the residuals of:

- Continuity equation
- Navier-Stokes momentum equations (r, θ, z components)

Residuals are calculated using autograd on reconstructed fields. Physics loss is scaled down:
```python
physics_loss /= 1e7
```

### 3. Combined Loss
```python
total_loss = data_loss + physics_loss
```

> 📌 Note: Physics loss is used **as a soft constraint**, guiding training without forcing convergence. This stabilizes learning without overwhelming the supervised signal.

---

## 🏋️ Training Entry Point

```python
if __name__ == "__main__":
    pinn = PINNSystem(hidden_dim=30, lr=1e-3, use_physics_batch=False)
    train_loss, test_loss = train_PINN(pinn, epochs=100, coord_batch_size=64)
```

Set `use_physics_batch=True` to use **full coordinate batches**, which improves accuracy but increases computational cost.

---

## 📈 Inference Example

```python
example_coords = torch.tensor([[0.1, 0.05, -0.1]], dtype=torch.float32)
condition = torch.tensor([630, 0.65])  # Unnormalized
result = pinn(example_coords, condition)
print(result)
```

Returns a dictionary of reconstructed physical fields (`P`, `Ut`, `Ur`, `Uz`) at the specified location.

---

## 🧪 Physics Residual Sanity Check

To test if residuals are reasonable:
```python
with torch.enable_grad():
    coords = torch.tensor(..., requires_grad=True)
    cond = torch.tensor(..., requires_grad=True)
    loss = pinn.compute_physics_residuals(coords, cond.unsqueeze(0))
```

---

## 📎 Notes

- Model is sensitive to the scaling of physical constants. If modifying the domain or units, re-check normalization.
- Setting `coords += 1e-6` is **crucial** to avoid `NaN` in autograd derivatives.
- Physics loss should not be minimized to zero; instead, it should remain bounded (e.g., < 1e2) for meaningful constraints.

---

## ✔ Result Discussion


## 📚 Future Plans

- Add support for additional physical quantities and alternative boundary conditions.
- Explore physics loss weighting via dynamic loss balancing (e.g., using uncertainty or GradNorm).
- If you have a better device (Mine is NVIDIA Geforce RTX 2060), you can use `use_physics_batch=True` to get a better model


---
