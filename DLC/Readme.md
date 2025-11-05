> Divice Replacement Note: The new experiments were runned on a Nvidia4060 GPU.

> To deploy the models, ensure to activate the `view.bat` in THIS FOLDER (temporarily unavailable)

> The previous attempts on other pump design, see in [here](../model_summon/) for more technical details

# Attemps to Make the Model Perform Better at Low Flow Rate Conditions

> During the presentation of our preliminary work at NURETH-21, a professor pointed out that the proposed SVD-based POD combined with a PINN regression model may not strictly qualify as a “Galerkin-POD” reduced-order model, since no explicit projection of the Navier–Stokes equations was performed. We fully acknowledge this distinction from the classical Galerkin framework. However, in complex geometries such as the LFR main coolant pump, performing a stable Galerkin projection is notoriously difficult due to the nonlinear modal interactions and numerical instability of the projected system. Instead, we adopted an SVD-POD approach that extracts dominant flow features in an energy-optimal manner, while employing the physics-informed neural network (PINN) as a nonlinear regression model to infer modal coefficients under varying operating conditions. The embedded physics constraints act as a weak regularization, guiding the data-driven regression toward physically consistent solutions without requiring explicit tensor projection. Although this approach departs from the “strict” Galerkin sense, it aligns with the modern hybrid ROM philosophy—trading exact projection for robustness and generalizability in realistic CFD applications.

## POD-PINN
In this work, the reduced subspace is constructed via SVD-based proper orthogonal decomposition (POD), and the modal coefficients are regressed by a physics-informed neural network (PINN).

Unlike the traditional Galerkin-projected POD-ROM that explicitly constructs low-dimensional governing equations, the present approach uses the physics constraints as a weak regularization to guide the data-driven regression process, thereby ensuring physical consistency while maintaining numerical robustness in complex geometries.

Although this approach departs from the strict Galerkin formulation, it aligns with the recent trend of hybrid data–physics reduced-order modeling that trades exact projection for robustness and generalizability in realistic CFD applications.

## Core Features
- `ReduceOrder.py`: Perform SVD-POD to the dataset, find the common, dominant modes of the flow field in the form of discrete vectors;
- `InterpolatorORI.py`: Modes interpolation, turn the discrete eigenvectors into a $\phi(x)$, by doing so we can apply loss functions with spatial derivatives;
- `ReconstructORI.py`: PINN training, to generate $\mathcal{N}(\beta;\theta)=a_i(\beta)$;
- `./ReducedResults`: where the modes are stored;
- `coordinates.csv`: the coordinates of the 0.5 span zone;
- `EXP.csv`: the dataset division;
- `ur.csv`, `ut.csv`, `uz.csv`, `P.csv`: the flow field distribution $\left[u_r, u_\theta, u_z, p\right]\left(x,\beta\right)$ on the 26 different operating conditions.

### Major Update on Oct. 25, 2025
- `ReduceOrder.py`: The demeaning step in the program has been updated. If the operating conditions only cause minor shifts in parameters (such as outlet flow or impeller speed) but the modes remain largely unchanged, the global mean can represent a "central/reference field," and the modes will explain the main shape differences around this common field. This resulting modes are easier to use for parameterization, coefficient interpolation, or constructing a unified ROM. Furthermore, this eliminates the need for regression to the mean, reduces the output dimensionality, and makes training easier.
- `InterpolatorORI.py`: Added index cache operation to avoid the re-indexing during the training. Meanwhile, the interpolation operations of the mean values are applied and the related interpolators are embedded into the mode interpolators to further optimize. 
- `ReconstructedORI.py`: Modified accordingly to suit for those changes. These 2 updates significant imporved the training speed of each epoch (decreased from 1.5s to about 0.5s per batch). Fixed some bugs and improved the interaction logic for file indexing and hyperparameter adjusting.
- Related `./ReducedResults` filefolder updated
