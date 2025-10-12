> To deploy the models, just copy all the `.pth` and `.png` files to the `.../model_summon/ReconstructORI` and activate `view.bat`.

# Attemps to Make the Model Perform Better at Low Flow Rate Conditions

> During the presentation of our preliminary work at NURETH-21, a professor pointed out that the proposed SVD-based POD combined with a PINN regression model may not strictly qualify as a “Galerkin-POD” reduced-order model, since no explicit projection of the Navier–Stokes equations was performed. We fully acknowledge this distinction from the classical Galerkin framework. However, in complex geometries such as the LFR main coolant pump, performing a stable Galerkin projection is notoriously difficult due to the nonlinear modal interactions and numerical instability of the projected system. Instead, we adopted an SVD-POD approach that extracts dominant flow features in an energy-optimal manner, while employing the physics-informed neural network (PINN) as a nonlinear regression model to infer modal coefficients under varying operating conditions. The embedded physics constraints act as a weak regularization, guiding the data-driven regression toward physically consistent solutions without requiring explicit tensor projection. Although this approach departs from the “strict” Galerkin sense, it aligns with the modern hybrid ROM philosophy—trading exact projection for robustness and generalizability in realistic CFD applications.

## POD-PINN
In this work, the reduced subspace is constructed via SVD-based proper orthogonal decomposition (POD), and the modal coefficients are regressed by a physics-informed neural network (PINN).

Unlike the traditional Galerkin-projected POD-ROM that explicitly constructs low-dimensional governing equations, the present approach uses the physics constraints as a weak regularization to guide the data-driven regression process, thereby ensuring physical consistency while maintaining numerical robustness in complex geometries.

Although this approach departs from the strict Galerkin formulation, it aligns with the recent trend of hybrid data–physics reduced-order modeling that trades exact projection for robustness and generalizability in realistic CFD applications.
