# Models Used in this Digital Twin

> the dataet remains the same as the NURETH21 paper 1193.

## Core Features
- `ReduceOrder.py`: Perform SVD-POD to the dataset, find the common, dominant modes of the flow field in the form of discrete vectors;
- `InterpolatorORI.py`: Modes interpolation, turn the discrete eigenvectors into a $\phi(x)$, by doing so we can apply loss functions with spatial derivatives;
- `ReconstructORI.py`: POD-PINN training, to generate $\mathcal{N}(\beta;\theta)=a_i(\beta)$;
- `NeuralOperators.py`: The FNO, CNO and [self-adaptive CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO), although did not interpreted in this paper, but we have realized its basic function in the practical engineering scenarios;
- `Encoders.py`: Due to the irregular mesh used in this research, there's need for NeuralOperators to encode the spatial information (MLP type & CNN type);
- `TRAIN_FullOrder.py`: The main training function of the FullOrder models. Support the DeepONet structure.
- `coordinates.csv`: the coordinates of the 0.5 span zone;
- `EXP.csv`: the dataset division;
- `ur.csv`, `ut.csv`, `uz.csv`, `P.csv`: the flow field distribution of the 0.5 span zone $\left[u_r, u_\theta, u_z, p\right]\left(x,\beta\right)$ on the 25 different operating conditions.

## POD-PINN
Most part remains the same as the method illustrated in the Nureth-Conference, but there are 2 main improvements:
1. The Interpolation Method: Used Pre-calculated interpolator to avoid training with the whole coordinates table, see in the improved `InterpolatorORI.py`. Significantly accelerated.
2. The Diffusion Term: In considering of the extermely small viscous coolant, we used the square of the 1st gradients to estimate 2nd gradients in order to imporve the landscape of the loss function and accelerate training.
```python
        # Simple viscosity approximation
        def simple_visc(u):
            """
            Approximate Laplacian of u for viscous term without second-order autograd.
            Uses sum of squares of first derivatives as rough estimate.
            """
            du = grad(u, coords)
            du_dr, du_dtheta, du_dz = du[:, 0:1], du[:, 1:2], du[:, 2:3]
            # Approximate Laplacian as sum of squared first derivatives
            lap_approx = du_dr ** 2 + (1 / r ** 2) * du_dtheta ** 2 + du_dz ** 2
            return lap_approx
```

## A Pure MLP
The very basic model for comparison. Nothing much to say.

## DeepONet
Allows us to use more complicated structure for solving the PDEs via the trunk network, for example [Self Adapted-CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO)

## Multi-input PINN (A modified DeepONet, didn't used in this papar)
> This model was first made public in my Chinese paper "Multi-input PINN Surrogate Model for Digital Twin of Lead-cooled Fast Reactor Main Pump. Atomic Energy Science and Technology, 59(5), 1085-1093.", which can be seen as a variant of the DeepONet (However, this thoughts was first originated from my undergraduate thesis in early 2024. At that time I just commenced the research roadmap of the digital twin of the LFR MCP, and didn't realize the whole business was actually FIND A FASTER PARAMETRIC PDE SOLVER WITH ACCEPTABLE ACCURACY. Therefor in that moment, to be honest, have no idea about operator learning.😓 Now I find the operator learning could be promising for this mission and proposed [Self Adapted-CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO)). I wish to change the trunk network into a Neural Operator network with better capability for solving PDEs in this paper.
