
# Reduced Order Modeling is not suitable enough for the problems with high non-linearity

## A Pure MLP
The very basic model for comparison

## DeepONet
Allows us to use more complicated structure for solving the PDEs via the trunk network, for example [Self Adapted-CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO)

## Multi-input PINN
> This model was first made public in my Chinese paper "Multi-input PINN Surrogate Model for Digital Twin of Lead-cooled Fast Reactor Main Pump. Atomic Energy Science and Technology, 59(5), 1085-1093.", which can be seen as a variant of the DeepONet (However, this thoughts was first originated from my undergraduate thesis in early 2024. At that time I just commenced the research roadmap of the digital twin of the LFR MCP, and didn't realize the whole business was actually FIND A FASTER PARAMETRIC PDE SOLVER WITH ACCEPTABLE ACCURACY. Therefor in that moment, to be honest, have no idea about operator learning.😓 Now I find the operator learning could be promising for this mission and proposed [Self Adapted-CFNO](https://github.com/LokimuKH19/SymPhONIC/tree/main/WhyWeakFNO)). I wish to change the trunk network into a Neural Operator network with better capability for solving PDEs in this paper.
