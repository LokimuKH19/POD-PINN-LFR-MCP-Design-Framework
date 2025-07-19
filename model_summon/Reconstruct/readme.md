# Result Discussion of POD-PINN with NN Interpolator

This is a detailed discussion of the enhanced program `../Reconstruct.py` (The new proposed method). It leverages pre-computed modal bases for the reconstruction of physical fields, and introduces a carefully balanced combination of **data loss** and **physics loss**.
# 📊 Comparison of NN-based and KNN-based Interpolation in PINN

## 📝 Summary
In our recent experiments with physics-informed neural networks (PINNs) combined with proper orthogonal decomposition models (POD), we compared two approaches for interpolating spatial modal functions: classic K-nearest neighbors (KNN) vs. neural network (NN)-based interpolation. While KNN is generally more accurate in direct interpolation tasks, the NN-based interpolator surprisingly resulted in comparable in the full PINN training process, especially when considering speed and gradient support.

## 🔍 Key Observations

### 🎯 1. Interpolation Accuracy
- 🟢 **KNN** gives more accurate predictions at known points.
- 🟡 **NN** interpolator introduces more approximation error on the dataset but is smooth and 2nd-ordered differentiable.

### 🔁 2. Gradient Flow
- ❌ KNN is non-2nd-differentiable, making it unsuitable for backpropagation through modal space. A shift must be done when computing the 2nd-ordered derivatives. 
- ✅ NN interpolation supports end-to-end gradient flow, which greatly improves smoothness of the trail function in PINNs.

### ⚡ 3. Training Efficiency
- 🚀 **NN-based interpolation** can run solely without datasets, leading to major speedups during training. Thus in the `../Reconstruct.py` I used more operating conditions (20) than `../ReconstructORI.py` (10) when computing physical losses. Also, the requirement of GPU memory became slighter.
- 🐢 KNN-based interpolation was computed with incorporating the whole `coordinates.csv`, which becomes a bottleneck in large-scale coordinate evaluations.

### 📉 4. Loss Behavior
- ⚠️ Physics residual loss with NN interpolation tends to be higher (due to the interpolation with pure NN might encounter issues like overfitting or something, though I have tried)  
- 📈 **Test loss actually goes down with Train Loss** in early training stage, indicating better generalization, as shown in the figure:

![POD-PINN with NN](./Losscurve_SEED42_LR0.005_HD30_HL2_Epoch113_WithPhysics1_WithBatch1.png)

❗The first 50 epochs are in the pre-training stage.

### ⚖️ 5. Overall Trade-off

| 🧩 Aspect                     | 🧭 KNN                     | 🚀 NN Interpolation         |
|------------------------------|---------------------------|-----------------------------|
| 🎯 Accuracy                  | ✅ Better (Train Loss 0.11, Test Loss 1.39)         | 🔸 Worse (the Least Test Loss 2.12 is given at the 35th epoch, with the Training Loss 0.84) |
| 🔁 Smoothness                | ↓ C1 on datapoints, C2 on others position                    | ↑ at least C2                       |
| ⚡ Training Speed            | 🐢 Slow (Based on the Interpolators with Dataset)       | ⚡ Significantly Faster  |
| 📉 Physics Residual          | ✅ Lower (×1e-7~1e-6 to normalize)                   | 🔸 Higher (×1e-17~1e-15 to normalize)           |
| 🧪 Generalization (comparing with Common NN whose test loss was 2.2)| ✅ Better         | ✅ Slightly Better   |

## ✅ Practical Conclusion
Although NN-based interpolation might not be as precise as KNN in direct interpolation tasks, the advantages in training speed, differentiability, and overall performance in PINN training make it a much better choice for practical purposes. It smooths out the training dynamics and provides gradients where needed, helping the model converge faster and generalize better.

> In the future, if the interpolators based on NN are enhanced this method has potential to replace the KNN interpolator based POD-PINN.

> I have tried similar ideas before, using the NNs to upgrade methods from linear algebras into a functional based one, see in [DRQI](https://github.com/LokimuKH19/DRQI).

> Instead of viewing POD as a purely linear algebraic decomposition on discrete data, reformulating it as a variational optimization problem over function space might be a promising way. This naturally leads to a neural-network-based approach, where the basis functions are parameterized and trained directly to minimize projection error — enabling more flexible, physically-informed, and generalizable continuous modal learning.

## 🧠 Combinining POD and Modal Interpolation with NN together?

In the previous version, we employed pure data-driven POD to extract main modals as eigenvectors. In specific, SVD is performed to snapshots matrix $X\in R^{m\times n}$ and the best **low-rank approximation** is obtained by combining the modals and the relative coefficients. However, these processes are still linear algebra tricks on **discrete matrix** and don't rely on the continuity of the practical physical system (or functional background), which therefore constraints its generalizability on physical modeling tasks.

🔁 What my idea is, regarding the POD as a **functional optimization problem**, which means searching for a series of **orthogonal function basis** in the Hilbert space and achieving the smallest projection error with the input functions. Possibly written as the following form?

```math
 \max_{\phi\in H} \int_\Omega \left(\int_T u(x,t)\phi(x) dx \right)^2 dt\quad\quad \rm{s.t.} \rm{norm}(\phi)_L^2 = 1
```

In this formula, $x$ refers to the spatial coordinates, $t$ refers to different snapshots. And these snapshots come the CFD simulation results.

This formula is an functional optimization problem, we only have to 

## 🔧 Key Idea

- Represent the **spatial mode function** \(\phi_\theta(x)\) using a neural network.
- Define a loss function to find the mode that captures the most energy:
  
\[
\mathcal{L}(\theta) = -\int_T \left( \int_\Omega u(x, t)\phi_\theta(x)dx \right)^2 dt + \lambda\left(\|\phi_\theta\|_{L^2}^2 - 1\right)^2
\]

- Replace the integrals with **Monte Carlo** approximations using sampled data points.
- Use **Adam** optimizer to train the network and get \(\phi_\theta(x)\).

---

## 🎯 What Do We Get?

We obtain:
- A **neural network approximation** of the most energetic mode \(\phi_\theta(x)\).
- No need for matrix decomposition — we’re solving a **functional extremum problem** directly.

---

## 📌 How to Compute Modal Coefficients?

Once we obtain \(\phi_\theta(x)\), we project the solution field \(u(x,t)\) onto it to get the time-dependent coefficient:

\[
\boxed{
\alpha(t) = \int_\Omega u(x, t) \cdot \phi_\theta(x) \, dx
}
\]

In code (discrete form):

```python
# coords: (M, D)
# u_vals: (M, T) -- each column is u(x) at time t
# phi_vals = phi_theta(coords)  # (M,)

alpha_t = torch.einsum('mt,m->t', u_vals, phi_vals) * dx
```

