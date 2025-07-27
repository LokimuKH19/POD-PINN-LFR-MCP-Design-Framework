# How to Use?

## 📁 Data Overview

In this folder, you’ll find the raw CFD data used for POD decomposition.

### 🧾 Files Dependency

- `coordinate.csv`  
  Contains the **polar coordinates** (`r`, `θ`, `z`) of all sampling points. These are taken from a thin cylindrical slice at `span = 0.65`, so the `r` values barely change — the variation mainly lies in `θ` (circumferential) and `z` (axial). [We haven't unlock the technology for performing SVD on a very large scale matrix, thus we only focus on several key spans which can represent the flow characteristics within the pump.]

- `P.csv`, `Ur.csv`, `Ut.csv`, `Uz.csv`  
  Each file corresponds to a physical quantity:
  - `P.csv` — Pressure  
  - `Ur.csv` — Radial velocity  
  - `Ut.csv` — Tangential velocity  
  - `Uz.csv` — Axial velocity
    Each **column** represents a specific operating condition (we have 27 total, as shown in `EXP.csv`), and each **row** corresponds to a coordinate point from `coordinate.csv`.

- `EXP.csv`
  Contains the operating conditions of the pump, each row represents a CFD result with different rotating speed $\omega$ (rpm) and outlet volumn flow rate $q_v$ (m3/s). The third column `test` refers to whether this result belongs to the testing set or not.

- For the dependency of the packages used in this repository, see in the [DRQI Repo](https://github.com/LokimuKH19/DRQI/blob/main/requirements.txt).

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
\mathbf{U}(\mathbf{x}, t) \approx \sum_{i=1}^{k} a_i(t) \, \phi_i(\mathbf{x}) + \bar{\mathbf{U}}(t)
```

Where:
    - $ϕ_i(x)$ are the spatial POD modes (i.e., flow patterns)
    - $a_i(t)$ are modal coefficients per condition
    - $\bar{U}(t)$ is the mean field at each snapshots $t$ — don’t forget this! Without it, your flow field will float in the void.

### 📁 What's in `ReduceResults/`

After running `ReduceOrder.py`, for each physical variable (`P`, `Ur`, `Ut`, `Uz`) represented as `*`, the following files will be generated in the filefolder `Interpolator`:

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

`Interpolation.py` is responsible for this step. To functionize the discreted $\phi_i(x)$ in the previous step.

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

Saves our sanity, and our folders. That's what we've done in the `Interpolation.py`.

This section documents the key insights and techniques that helped us consistently reduce modal interpolation error to the **1e-4 MAE level**,however, the original main modal values ranged between **1e-7 to 1e-2**. This translates to a larged relative error, which is hard to accept (Though the Pearson Correlation Factor is mostly larger than 0.99). Still, we found these modifications has potential to enhance the interpolation model's performance.

---

### ✅ Key Observations

#### 1. Depth over Width: Why `[20, 20, 20, 20]` Works Better Than `[64, 64]`

While `[64, 64]` has more total parameters (4096), a deeper network like `[20, 20, 20, 20]` (1200 parameters) tends to generalize better and escape local minima more easily.

- **Deeper networks** offer more compositional power.
- **Fewer parameters** make it less prone to overfitting.
- The "depth" often helps capture hierarchical structure in modal behavior.

---

#### 2. Use of `ReduceLROnPlateau` Scheduler

We apply a learning rate scheduler to **automatically reduce the learning rate** when the test loss plateaus. This improves convergence and fine-tuning performance.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    model.optimizer, mode='min', factor=0.6, patience=30, verbose=True
)
```

- This avoids being stuck in a sub-optimal local minimum.
- Learning rate drops are triggered **only when necessary**, not every N epochs.

---

#### 3. Batch Size Matters

We found that a batch size of **128** strikes a good balance:

- Smaller batches introduce too much noise.
- Larger batches lead to smoother but less exploratory updates.

Batch size influences gradient variance, which helps the model escape shallow local minima.

---

When Activate `Interpolator.py`, neccesary model parameters and normalizers will be saved automatically in the `Interpolator` filefolder. See the calculation settings in the `Reproductive Description.txt` in that folder.


## 🧩 What does the Original Interpolator Look Like?

### 🗺 Background

In our original research paper, we adopted a 3-nearest-neighbor interpolation combined with a Gaussian kernel weighting scheme to ensure the interpolation result is smooth and differentiable. This approach was chosen to provide a balance between interpolation accuracy and smoothness, crucial for downstream applications involving gradients and higher-order derivatives, such as Physics-Informed Neural Networks (PINNs) used later.
However, the initial implementation lacked a data pre-loading (pre-reading) mechanism, which led to slower loss function computations during training due to repeated disk I/O or re-computations. What's more, we are unsatisfy with the performance of the interpolators based on neural networks (see in the `Interpolator` filefolder). These are the motivation which drive me to propose a better performed one, with accuracy and significantly improved speed.

### 🔑 Key Improvements in This Implementation
#### 1) Interpolation Method and Accuracy
- We use k-nearest neighbors (k=3 by default) to localize interpolation and avoid global smoothing artifacts.
- A Gaussian kernel weight based on neighbor distances provides a smooth weighting that enables differentiability.
- Exact sample points may cause issues with computing second derivatives because interpolated values exactly match sample values, often resulting in non-smooth or undefined second derivatives.
- To mitigate this, a tiny random perturbation (epsilon shift) is added to inputs before interpolation when computing gradients. This helps avoid hitting the exact sample points where higher-order derivatives are unstable or undefined.
#### 2) Points Used for PINN Training
- PINNs require the evaluation of residuals (losses) at arbitrary spatial points.
- Since our interpolation function is continuous and differentiable (except at exact sample points), we can generate any training points on-the-fly for the PINN.
- However, to ensure well-defined second derivatives, we avoid evaluating exactly at sample points by adding a minuscule perturbation (on the order of 1e-6~1e-8). If you don't do that, you will get a tensor full of 'nan's.
- This subtlety is critical to maintain numerical stability and gradient availability in PINN loss computations.
#### 3) Efficiency Considerations
- The interpolator is instantiated once and treated as a global object during training.
- This avoids repeated file reads and re-computation of interpolation weights.
- The batch operations in PyTorch ensure GPU acceleration can be leveraged for fast interpolation and gradient computations.
- Adding the perturbation noise is very cheap computationally but yields significant benefits in gradient stability.

This design of this improved KNN Interpolator is in `InterpolatorORI.py`.

## 🎥 Now the modes become C2 functions. So PINN Construction is continued
### 2 ways to construct PINN for regress the modal coefficients:
- `ReconstructORI.py`: PINN with KNN interpolators
- `Reconstruct.py`: PINN with NN interpolators
- The Evaluation of the 2 models, see in the filefolders with the same names.

In both cases, the PINN only regresses the **modal coefficients**, and the full flow field is reconstructed using pre-interpolated modal basis functions.

### 📌 Governing Equations

We define the residuals of the control equations in the blade reference frame for use in the loss function construction. These include the continuity equation and momentum equations (radial, tangential, axial), with centrifugal and Coriolis effects included. Also see in the paper. 

⚠ Due to the NS equations used in our paper is defined on the rotator, therefore we must use the **relative velocities** in the following loss functions. In particular, the $u_\theta$ here should be $u_\theta-\omega\cdot r$ in the absolute reference system.

⚠ The rotating direction of the impeller is clockwise (as the z direction selected along the coolant flow in our paper), therefore negative $\omega$ is used instead. In addition, the unit `rpm` needs to be transformed into `rad/s` before caculating the physical losses.

⚠ The Laplacian Operator in cylindrical coordinates should be: $\nabla^2 f = \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial f}{\partial r}\right)+\frac{1}{r^2}\frac{\partial^2 f}{\partial r^2}+\frac{\partial^2 f}{\partial z^2}$ (Though the vicous terms can be ignored for the liquid metal, and even if you made mistakes here it won't influence the results significantly, I strongly advise you to check this when you do similar things. Just in case the fluid suddenly becomes water or oil.)

- The Continuum Equation:
```math
F_C(u_r,u_\theta,u_z)=\frac{1}{r}\frac{\partial(ru_r)}{\partial r} + \frac{1}{r}\frac{\partial u_\theta}{\partial \theta}+\frac{\partial u_z}{\partial z}=0
```
- The r-Momentum Equation:
```math
F_R(u_r,u_\theta,u_z,p)=u_r\frac{\partial u_r}{\partial r} + \frac{u_\theta}{r}\frac{\partial u_r}{\partial \theta}+u_z\frac{\partial u_r}{\partial z}-\frac{u_\theta^2}{r}+\frac{1}{\rho}\frac{\partial p}{\partial r}-\frac{\mu}{\rho}\left(\nabla^2 u_r-\frac{u_r}{r^2}-\frac{2}{r^2}\frac{\partial u_\theta}{\partial \theta}\right)-\omega^2 r+2\omega u_\theta=0
```

- The θ-Momentum Equation:
```math
F_Θ(u_r,u_\theta,u_z,p)=u_r\frac{\partial u_\theta}{\partial r} + \frac{u_\theta}{r}\frac{\partial u_\theta}{\partial \theta}+u_z\frac{\partial u_\theta}{\partial z}+\frac{u_\theta u_r}{r}+\frac{1}{\rho r}\frac{\partial p}{\partial \theta}-\frac{\mu}{\rho}\left(\nabla^2 u_\theta-\frac{u_\theta}{r^2}+\frac{2}{r^2}\frac{\partial u_r}{\partial \theta}\right)-2\omega u_r=0
```

- The z-Momentum Equation:
```math
F_Z(u_r,u_\theta,u_z,p)=u_r\frac{\partial u_z}{\partial r} + \frac{u_\theta}{r}\frac{\partial u_z}{\partial \theta}+u_z\frac{\partial u_z}{\partial z}+\frac{1}{\rho}\frac{\partial p}{\partial z}-\frac{\mu}{\rho}\nabla^2 u_z+g=0
```

### 🔍 Notes on Weak Form and Domain Constraints
In this work, we primarily focus on a localized modeling strategy based on reduced-order representations. Since our surrogate models are trained in a low-dimensional parametric space, the corresponding reconstruction is only valid within a spatially bounded region. This inherently restricts the direct application of boundary conditions in the classic PDE sense. And in `Reconstruct.py` I also attempt to use the mean value as another constraint, however I finally gave up it due to the high complexity of calculating the integral of the whole region.

As a result, when physical constraints are introduced, we opt for enforcing them in a strong form — meaning they are applied pointwise (or approximately so) via physics-informed losses, rather than through variational integration.

This choice is motivated both by practicality (reduced cost) and necessity (no access to full-domain information or Dirichlet boundaries). However, we recognize that weak formulations — where physical laws are enforced in an integral sense over test functions — offer powerful benefits, especially in aligning with classical finite element methods (FEM).

To illustrate this, we plan to include a clean and minimal example of a weak-form PINN in this repository. Specifically, we will use a standard benchmark problem: the 2D lid-driven cavity flow.

In that simplified setting, we can define a full domain and boundary conditions, allowing us to clearly demonstrate how to:

- Construct a weak-form loss based on FEM principles

- Integrate over the spatial domain using quadrature points

- Compare the performance and flexibility with strong-form PINNs

Stay tuned for updates in the `WeakForm/` folder, where the lid-driven cavity demo will be later included. This supplementary example will serve as a reference for those interested in more advanced PINN formulations — especially those that blend well with traditional numerical schemes like Galerkin FEM.

## 💻 Using the Model

I developed a UI in `UserInterface.py` based on `streamlit`. If you don't have it on your device, please enter the cmd console and enter:

```cmd
pip install streamlit==1.44.1
```

What you can do to analyze the mode are:

- Run `UserInterface.py` by double clicking `view.bat`;
- Enter [http://26.26.26.1:8501](http://26.26.26.1:8501) in your browser and jump to the UI;
- Load model from `Reconstruct` and `ReconstructORI` filefolder, as displayed in the left of the window;
- Enter `omega` and `qv` according to the hints and press `Comput Flow Field`;
- Obtain the flow field and other model information in the interface.

The screenshots for this App:

![screenshot1](./screenshot1.png)

![screenshot2](./screenshot2.png)

### 🔍 Overall Design and Error Analysis Strategy

This UI performs detailed **prediction evaluation** of a surrogate physical model (ROM-PINN or others) against CFD or experimental data. If you entered working conditions in the dataset `EXP.csv`, the following error analysis will be automatically carried out.The evaluation consists of two complementary components:

---

### 🧷 Point-to-Point Error Distribution (Local View)

The raw relative error of the predicted physical quantity (Pressure and the 3 velocities Ur, Ut, Uz) is calculated point-by-point in `coordinate.csv` and then displayed, to show the spatial error distribution directly.

---

### 🧷 Point-to-Point Error Statistics (Detialed View)

To understand **localized model behavior**, this mode visualizes the **distribution of pointwise relative error**.

Each physical quantity (Pressure, Ur, Ut, Uz) is evaluated using:

- **CDF (Cumulative Distribution Function)**: Plots the proportion of points with error below a certain value.
- **Error Histogram (Ratio-based)**: Displays error density distribution as a bar chart overlaid on the CDF curve.
- **Display Cut-off Slider**: Controlled by the user (default 90%), this limits the x-axis of the plots to focus on **majority behavior** while **excluding extreme tails** caused by:
  - Mesh noise,
  - CFD data artifacts,
  - Discretization error from structured-unstructured conversion.

> 🧠 Example: Set the "Error Display Cut-off Percentile" to 5% means **zooming into the most significant 95% of the data**, filtering out the worst outliers (usually <5%) to ensure the statistical figures readable.

---

### 🧮 Overall Error Analysis (Global View)

This view provides **global error metrics** to reflect model performance over the entire flow domain. It includes:

- **Area-Weighted Relative Error**: Accounts for irregular sampling point distributions by applying Voronoi-based area weights.
- **Global Normalization**: Normalization of both predicted and true values uses only `y_true` min/max to ensure stability and physical interpretability.
- **Masked Small Values**: Regions with negligible values (e.g., velocity near zero) are masked using a threshold (e.g., 0.01) to eliminate noise artifacts.

> 📌 This view helps in judging **how good the model is on average** while avoiding over-reaction to extreme local outliers.

---

### 🧵 Complementary Insight

- While **overall metrics** (like MSE or mean relative error) give a **summary score**, they can be misleading if a small fraction of points have extreme errors.
- On the other hand, **point-to-point statistics** reveal **error structure** — e.g., whether the model consistently overestimates pressure, or performs better in core flow vs. boundary regions.
- Thus, these two views **mutually reinforce** the interpretation:
  - Use global metrics to compare model versions or hyperparameters.
  - Use pointwise plots to debug physical inconsistencies and spatial generalization.
> ✨ This evaluation framework is especially useful in scientific ML scenarios where **physical plausibility** and **geometric fidelity** matter as much as raw accuracy.
---

### ⚙️ Key Techincal Details

| Item                         | Description                                                    |
|----------------------------------|----------------------------------------------------------------|
| Error Display Cut-off Percentile of Point-to-Point Error Statistics | Controls how much of the CDF and histogram x-axis is displayed (default: 90%). |
| Area Weighting when computing the  overall relative error | Internally applies Voronoi area to each point to account for non-uniform mesh distribution. |
| Normalization process when computing the overall relative error | Applied using true values only, ensuring consistent scale across evaluations. |
| Value Masking Threshold | Filters values below 0.01 in the overall error, and 0.2m/s in the point-to-point error (the level of the velocity is 1e1 in the dataset) to avoid numerical instability and artifacts. |

---


### 📝 Notes

- DO NOT activate `UserInterface.py` directly though python. A UI based on `streamlit` can only be started by cmd command `streamlit run UserInterface.py` at the local directory (as shown in `view.bat`).
- The UI allows real-time adjustment and visualization — Thanks to Streamlit’s auto-reload, the browser's next action will be constantly following the latest version of `UserInterface.py`.
- DO NOT refresh the page. Some serious problem related to the index of the dependency files would occur. If that happens you can only stop the view.bat and re-run it.


## 😺 How well does our model perform

The "model" discussed in this section refers to the best-performing surrogate:  
`.ReconstructORI/Checkpoint_SEED42_LR0.01_HD30_HL2_Epoch63_WithPhysics1_WithBatch1_Pretrain50.pth`

In [`./Reconstruct/readme.md`](./Reconstruct/readme.md) and [`./ReconstructORI/readme.md`](./ReconstructORI/readme.md), we already explored its **generalization ability** by examining convergence curves. The synchronized convergence between training and validation underlines the effectiveness of **physics-informed supervision**, which not only regularizes the network but ensures physical consistency.

---

### 🚀 Inference Efficiency & Practicality

This surrogate model can predict **physical fields (P, Ur, Ut, Uz)** at **7,999 spatial points** almost instantly for any given working condition, representing a **dramatic speed-up compared to traditional CFD simulations** (Training a model=Doing infinite traditional CFD simulations).

Moreover, thanks to the nature of **PINNs**, this model inherently suppresses high-frequency numerical noise found in CFD data by solving the underlying control equations.

---

### 📈 Interpreting the Error

While **relative error** is used as a primary metric, it must be interpreted with caution:

- CFD data itself contains numerical error and mesh-related artifacts.
- Our model is not just regressing CFD output but solving physical laws.
- However, **consistently high relative errors still indicate model flaws** or data inadequacies.

---

### ✅ Summary of Model Performance

- Performs best under conditions where **omega ≥ 0.27** (medium to high flow rate).
- Among the primary quantities of interest in axial pumps:
  - **Pressure (P)**
  - **Tangential Velocity (Ut)**
  - **Axial Velocity (Uz)**  
  The model maintains **area-weighted relative error mostly under 15%**, rarely exceeding 35%.
- **Similarity scores** (Pearson correlation) are often above 80%.
- **Error concentration** tends to appear:
  - Near blade regions with complex geometry (regardless of training/test split).
  - In some low-flow-rate conditions, errors can rise to ~40%.
  - Occasionally in off-blade regions — may relate to underrepresented flow dynamics.

> 📉 This best model in the repo was trained using same architecture ([2, 30, 30, 5]) in our paper also **struggle with small-flow-rate predictions**, though they used extrapolation, while this repo uses interpolation to enhance performance on small-flow conditions. It seems that the division of training/testing set has not siginificant influence on the final performance of the model, also caused by PINN's nature. In addition, on the dataset in this repo, the testing loss POD-PINN 1.35 has not absolute advantage comparing to POD-PureNN's 2.20 (at the same level), causing similar performace of each convergenced model. 

---

### 🧠 Explanation

A likely reason for reduced performance in small flow-rate regimes is **insufficient POD modes**. Under low flow conditions, axial pumps exhibit **more complex and unstable flow features**, requiring a **higher number of modes** to accurately reconstruct the true flow field. Actually, using 95% energy propotion to perform order reduction on a highly unlinear system (driven by NS equation) is probably insufficient. When the flow rate is relatively low, the dead zone may possibly occur within the near-blade region, leading to the low velocity magnitude (~0m/s). In this occasion the relative error becomes inappropriate due to divided by zero.

This underscores a key challenge in reduced-order modeling: **balancing compression with expressive fidelity**.
