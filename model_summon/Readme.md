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

`ReduceOrder.py` is responsible for the Order Reduction process. Due to there's no random process, just simple linear algebra skills in this step, there's no need to use random seeds.

The algorithm used here is **Proper Orthogonal Decomposition (POD)** — a classic linear method to identify dominant flow structures.

### 🔍 How it works (in plain words):

We take a bunch of CFD snapshots (each column in the data = one condition), subtract the mean to center the data, and then:
- 🧮 Compute the covariance matrix
- 📉 Perform eigen-decomposition
- 🧲 Pick the top `k` modes that capture most of the energy
- ✨ Project the original data onto those modes to get the **modal coefficients**

The flow field is then approximated as:

```math
\mathbf{U}(\mathbf{x}, t) \approx \sum_{i=1}^{N} a_i(t) \, \phi_i(\mathbf{x})
```
