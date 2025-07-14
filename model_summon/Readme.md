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
