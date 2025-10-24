> To deploy the models, ensure to activate the `view.bat` in THIS FOLDER (temporarily unavailable)

> The previous attempts on other pump design and physical properties, see in [here](../model_summon/) for more technical details

# Attemps to Make the Model Perform Better at Low Flow Rate Conditions

> During the presentation of our preliminary work at NURETH-21, a professor pointed out that the proposed SVD-based POD combined with a PINN regression model may not strictly qualify as a “Galerkin-POD” reduced-order model, since no explicit projection of the Navier–Stokes equations was performed. We fully acknowledge this distinction from the classical Galerkin framework. However, in complex geometries such as the LFR main coolant pump, performing a stable Galerkin projection is notoriously difficult due to the nonlinear modal interactions and numerical instability of the projected system. Instead, we adopted an SVD-POD approach that extracts dominant flow features in an energy-optimal manner, while employing the physics-informed neural network (PINN) as a nonlinear regression model to infer modal coefficients under varying operating conditions. The embedded physics constraints act as a weak regularization, guiding the data-driven regression toward physically consistent solutions without requiring explicit tensor projection. Although this approach departs from the “strict” Galerkin sense, it aligns with the modern hybrid ROM philosophy—trading exact projection for robustness and generalizability in realistic CFD applications.

## POD-PINN
In this work, the reduced subspace is constructed via SVD-based proper orthogonal decomposition (POD), and the modal coefficients are regressed by a physics-informed neural network (PINN).

Unlike the traditional Galerkin-projected POD-ROM that explicitly constructs low-dimensional governing equations, the present approach uses the physics constraints as a weak regularization to guide the data-driven regression process, thereby ensuring physical consistency while maintaining numerical robustness in complex geometries.

Although this approach departs from the strict Galerkin formulation, it aligns with the recent trend of hybrid data–physics reduced-order modeling that trades exact projection for robustness and generalizability in realistic CFD applications.

## Core Features
- `ReduceOrder.py`: Perform SVD-POD to the dataset, find the common, dominant modes of the flow field in the form of discrete vectors;
- `InterpolatorORI.py`: Modes interpolation, turn the discrete eigenvectors into a $\phi(x)$, by doing so we can apply loss functions with spatial derivatives;
- `ReconstructORI.py`: PINN training, generate $\mathcal{N}(\beta;\theta)=a_i(\beta)$;
- `./ReducedResults`: where the modes are stored;
- `coordinates.csv`: the coordinates of the 0.5 span zone;
- `EXP.csv`: the dataset division;
- `ur.csv`, `ut.csv`, `uz.csv`, `P.csv`: the flow field distribution $\left[u_r, u_\theta, u_z, p\right]\left(x,\beta\right)$ on the 26 different operating conditions.

### Update on Oct. 23, 2025
Several Improvemment has been applied to the POD-PINN model, we added the normalization into the physical losses, therefore the scaling factor of the model can be removed:

```python
        # Using the normalized Phy_loss
        FC = (FC-torch.min(FC))/(torch.max(FC)-torch.min(FC))
        FR = (FR-torch.min(FR))/(torch.max(FR)-torch.min(FR))
        FTH = (FTH-torch.min(FTH))/(torch.max(FTH)-torch.min(FTH))
        FZ = (FZ-torch.min(FZ))/(torch.max(FZ)-torch.min(FZ))
```
Consequently, the monitor of the physical loss is also updated.

# Reduced Order Modeling is not suitable enough for the problems with high non-linearity

## A Pure MLP
The very basic model for comparison

## DeepONet
Allows us to use more complicated structure for solving the PDEs via the trunk network, for example [Self Adapted-CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO)

## Multi-input PINN
> This model was first made public in my Chinese paper "Multi-input PINN Surrogate Model for Digital Twin of Lead-cooled Fast Reactor Main Pump. Atomic Energy Science and Technology, 59(5), 1085-1093.", which can be seen as a variant of the DeepONet (However, this thoughts was first originated from my undergraduate thesis in early 2024. At that time I just commenced the research roadmap of the digital twin of the LFR MCP, and didn't realize the whole business was actually FIND A FASTER PARAMETRIC PDE SOLVER WITH ACCEPTABLE ACCURACY. Therefor in that moment, to be honest, have no idea about operator learning.😓 Now I find the operator learning could be promising for this mission and proposed [Self Adapted-CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO)). I wish to change the trunk network into a Neural Operator network with better capability for solving PDEs in this paper.
