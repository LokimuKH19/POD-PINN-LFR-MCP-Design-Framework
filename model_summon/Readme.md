# How to Use?

## 📁 Data Overview

In this folder, you’ll find the raw CFD data used for POD decomposition.

### 🧾 Files

- `coordinates.csv`  
  Contains the **polar coordinates** (`r`, `θ`, `z`) of all sampling points. These are taken from a thin cylindrical slice at `span = 0.65`, so the `r` values barely change — the variation mainly lies in `θ` (circumferential) and `z` (axial).

- `P.csv`, `Ur.csv`, `Ut.csv`, `Uz.csv`  
  Each file corresponds to a physical quantity:
  - `P.csv` — Pressure  
  - `Ur.csv` — Radial velocity  
  - `Ut.csv` — Tangential velocity  
  - `Uz.csv` — Axial velocity  

  Each **column** represents a specific operating condition (we have 25 total), and each **row** corresponds to a coordinate point from `coordinates.csv`.

### 🌀 Symmetry Assumption

Because the blade geometry is **periodic**, the flow field also shows symmetry.  
So instead of analyzing the whole annular flow passage, we can safely "cut a cake slice" — just one blade passage — and still capture the key physics. This is what allows us to simplify things and focus on a local region. This can allow us to send all the neccessary data into the CPU of a common Laptop once in for all to do the Order Reduction.

---

✏️ Think of this dataset as a snapshot of how the flow behaves near one blade at mid-span. The goal is to reduce this high-dimensional, multi-condition flow into a small number of dominant modes — so we can later *learn* and *predict* flow behavior under new conditions much faster than traditional CFD.
## ✂ Reduce Order Modeling

`ReduceOrder.py` is responsible for the Order Reduction process. Since it's purely based on linear algebra (no randomness involved), there's **no need to worry about random seeds** at this stage.

---

### 🔍 How it works (in plain words):

We take a bunch of CFD snapshots (each column in the data = one operating condition), subtract the mean to center the data, and then:

- 🧮 Compute the covariance matrix  
- 📉 Perform eigen-decomposition  
- 🧲 Pick the top `k` modes that capture most of the system's energy  
- ✨ Project the original data onto those modes to get the **modal coefficients**
- 📚 Also see the illustration in the paper Section 2.3.

This gives us a low-dimensional approximation of the original flow field:

```math
\mathbf{U}(\mathbf{x}, t) \approx \sum_{i=1}^{k} a_i(t) \, \phi_i(\mathbf{x}) + \bar{\mathbf{U}}(\mathbf{x})
```

Where:
    - $ϕ_i(x)$ are the spatial POD modes (i.e., flow patterns)
    - $a_i(t)$ are modal coefficients per condition
    - $\bar{U}(x)$ is the mean field — don’t forget this! Without it, your flow field will float in the void.

### 📁 What's in `ReduceResults/`

After running `ReduceOrder.py`, for each physical variable (`P`, `Ur`, `Ut`, `Uz`) represented as `*`, the following files will be generated:

| File | Description |
|------|-------------|
| `modes_*.csv` | Spatial POD modes. Each column is a mode (from 1 to k). Think of these as the “flow patterns.” |
| `coefficients_*.csv` | Modal coefficients per working condition (i.e., time or snapshot index). Each row = one condition. |
| `eigvals_*.csv` | Eigenvalues of each mode, representing their energy contribution. |
| `mean_*.csv` | The mean flow field. ⚠️ This must be added back during reconstruction, or the result will be centered incorrectly. |
| `EnergySort_*.png` | Bar plot showing how much energy each selected mode contributes — helps you decide if your k is “good enough.” |
| `reconstructed_*.csv` | The final reconstructed result (mean + modal sum). You can compare this to the original CFD data. |

This directory is basically your **low-dimensional version of the full CFD results**, ready to be used for machine learning modeling, fast estimation, or sanity checks.

👉 You might notice that the **Mean Squared Error (MSE)** shown in the console is quite large for velocity (especially `U`), while relatively small for other quantities. That’s because we **did not normalize** any field before POD — which is actually fine here, since **POD is linear** and we don’t strictly need normalization unless doing some further nonlinear work.  
Still, if you check the **correlation coefficients**, they perform nicely (~1), which means the **flow patterns and trends are well captured** even if absolute magnitudes aren’t perfect.

👀 Now, what comes next *diverges* a bit from what the paper did — in the paper, we used a **nearest-neighbor interpolation** method to estimate modal coefficients for arbitary position. And calculated the derivatives of each points in advance before PINN training. It worked, but honestly, it's a bit... primitive.

Here, we're taking it a step further by building a smoother, more flexible, and **learnable interpolator**. For:
- 💨 **Performance**: Once trained, neural interpolators can generalize to new cases much faster than retracing nearest neighbors from a large dataset.
- 🎯 **Precision**: However, the model might encounter underfitting issues than the 3-nearest neighbor one.
In short — smarter, faster, better. Let's get building.

## 🧩 Interpolator Construction

`Interplotation.py` is responsible for this step. To functionize the discreted $phi_i(x)$ in the previous step.

### 🧠 What's the plan?

Now, let’s talk about how we want to build the interpolator.

Take a look at the `coefficients_*.csv` files — each one contains modal coefficients for a specific physical quantity (P, Ur, Ut, Uz). Each column is a mode (like `Mode 1`, `Mode 2`, etc.), and each row corresponds to a specific simulation case, with a coordinate that has three components: **r**, **θ**, and **z**. So this sets us up with a nice little problem:

> Learn a mapping from `(r, θ, z)` ⟶ `(φ₁, φ₂, φ₃, φ₄)` for each variable.

That’s a **3 → 4 regression task**, and now the question becomes:  
Do we want to:

- 🧩 Train **one big neural network per field** to output all 4 modal coefficients at once?  
  ✅ Pros: Fewer models to manage, easier training pipeline, potentially shared feature learning.  
  ❌ Cons: Slightly harder to tune, may need more complex architecture.

**OR**

- 🔬 Train **four separate networks per field**, one for each mode?  
  ✅ Pros: Potentially easier to debug and tune per mode.  
  ❌ Cons: More models (16 in total!), more files, more chaos in general.

Personally, we’re leaning toward the **“one network per field”** route. That gives us **4 networks total**, one for `P`, `Ur`, `Ut`, and `Uz` — each mapping `(r, θ, z)` to a 4-dimensional output.

Saves our sanity, and our folders. That's what we've done in the Interplotation.py.
