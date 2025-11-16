# Models Used in this Digital Twin

> the dataet remains the same as the NURETH21 paper 1193.

- Conclusion: Kind of counter-intuition. Because the Reduced Order model does not require generalization to the solution space, its solution space is significantly smaller compared to the full-order model, making it easier to obtain high-accuracy results.

- In fact, in the exploration of full-order surrogate models, we explored not only the most basic DeepONet structure reported in the paper, but also pure MLP, but the results were unsatisfactory. The errors of stress and Uz remained high, and the correlation was not significantly improved. In order to improve the generalization ability of spatial coordinates, we also tried **DeepONet** with **CNN**, **Transformer**, **FNO**, **CNO**, and our independent tech [**self-adaptive CFNO**](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO) with CNN/MLP coordinate encoders as the backbone network, as well as modified DeepONet in a Chinese paper that I wrote a long time ago (however, the solution format of that paper was rectangular coordinates instead of polar coordinates. In addition, the biggest difference was that the paper used more than 160 sets of simulation conditions as the dataset, and the data point sampling density was much higher than that of this paper. Overall, the sampling of the condition parameters obtained barely satisfactory results, but it is obvious that this process still relies heavily on traditional CFD operations). In summary, these surrogate models actually perform worse than the reduced-order model POD-PINN on sparse datasets.

- I used to think that the Reynolds number (Re) for this problem was very high and the nonlinearity was very large, however, a week ago when I was providing a technical consultation I simulated a experimental device which measures the relationship between the flow velocity and the corrosion effecs of the LBE. The result reminded me that, the kinematic viscosity of LBE is very small, leading to the minor influence of the sheer stresses, and therefore may not form very obvious vortices. For example, if you only have a cylinder rod, and just stick it into very hot LBE and then rotate, you'll find only the fluid very near the wall would be influnced and most of the velocity is contributed by the tangential component. In contrast, the majority of the flowfield would remain stable. That's why we use a pump with blade to drive the LBE because it's difficult to move by "twisting", but it could be "pushed" (Normal stress dominates the flow field). In other words, the flow field of LBE (and other liquid metals in this category) exhibits, to some extent, the characteristics of the **potential flow**. Perhaps linear models are not as bad as imagined in this kind of tasks.

- Our nest plan: The planning SymPhONIC's numerical method should employ Galerkin-POD-based finite element simulation for this modal extraction process, thus avoiding the overfitting problem inherent in the purely data-driven POD step (this paper, due to its purely data-driven approach, also learned dataset bias, resulting in poor performance under low flow conditions). Regarding potential optimization directions for the full-order model, since the vast majority of errors are concentrated at the blade boundaries, introducing hard constraints can significantly improve the error tendency of velocity distribution. However, this relies on a precise mathematical description of the blade profile, and embedding accurate complex geometric profiles into the network training does pose a challenge. In short, the blade shape description needs to be embedded into the network output. However, the error in pressure prediction in the full-order model is uniformly distributed across the entire solution domain. Considering the inherent relativity of pressure in fluid simulation, adding hard pressure constraints at the inlet and outlet might help improve pressure accuracy.


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
- `ResultReading.py`: Streamlit code for the model analyzer
- `view.bat`: Start the Streamlit analyzer UI
- `./ReducedResults`: Includes the extracted POD modes and the corresponding coefficients dataset;
- `./ReconstructORI`: Includes all the POD-PINN models illustarted and not illustrated in the paper, and a simple analysis code used in the Fig. 8 `./ReconstructORI/analysis.py`;
- `./FullOrderReconstruct`: Including all Full Order surrogate models reported and not reported in the paper, and a simple analysis code used in the Fig. 12 `./FullOrder/analysis.py`.
- `./results`: The default directory for the model analyzer UI to store the summary csv files. Includes an analysis code for the evaluation of the model on the whole dataset, as used in the Fig. 10.

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
