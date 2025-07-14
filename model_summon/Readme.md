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
So instead of analyzing the whole annular flow passage, we can safely "cut a cake slice" — just one blade passage — and still capture the key physics. This is what allows us to simplify things and focus on a local region.

---

✏️ Think of this dataset as a snapshot of how the flow behaves near one blade at mid-span. The goal is to reduce this high-dimensional, multi-condition flow into a small number of dominant modes — so we can later *learn* and *predict* flow behavior under new conditions much faster than traditional CFD.
