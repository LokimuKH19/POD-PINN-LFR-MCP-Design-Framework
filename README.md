# 🌀 POD-PINN Based Main Pump Flow Field Predictor

## What's this thing even for?

This repo is for predicting the flow field of a **lead-cooled fast reactor main pump** under different operating conditions — quickly. Like, way faster than rerunning CFD every single time. The idea is: get the trends, save your time, and maybe (just maybe) shorten the design loop in the early stages.

## Compared against what?

The model was trained and tested using good ol’ CFD results — **Ansys CFX** simulations, to be exact. It's not just guessing in the dark.

## So... how well did it do?

Okay, we’ll be honest — not perfect. We’re dealing with **very high Reynolds numbers**, and we only used **4 POD modes**. That captured about 90% of the energy, but turns out, 90% isn't enough when the flow gets wild.

Why the error?  
- High nonlinearity, few modes = still a trade-off between speed and accuracy  
- Reynolds numbers are huge → behavior becomes chaotic  
- The method’s better at catching **flow trends** (correlation ≈ 1), but struggles with absolute precision

In short: **great for trend detection, not so great for exact numbers**. But hey, sometimes that's all you need.

## Wait, the code doesn't match the paper?

Yep. The paper was based on earlier results — and, well, **conference page limits are brutal**. So we had to cut a lot of the juicy details. What you see here is a more refined version. Consider this the “director’s cut.”

---

## How does it work?

We combined two fancy techniques:

### 1. POD + PINN = ❤️
- **POD** (Proper Orthogonal Decomposition) breaks down the CFD flow fields into modes.
- Then **PINNs** (Physics-Informed Neural Networks) try to regress the modal coefficients under different boundary conditions.

**Equations?**
- POD:  
  $$ \mathbf{U}(\mathbf{x}, t) ≈ \sum_{i=1}^{N} a_i(t) \phi_i(\mathbf{x}) $$
- PINN:  
  Neural networks trained with physical constraints (governing equations baked into the loss function).

### 2. Radial layering (a.k.a. how we avoided running out of RAM)
- Initially, we ran POD **layer by layer** along the radial direction (cylindrical slices).
- But this caused issues when trying to interpolate the modal coefficients in the **r** direction.
- So the paper only reported **layer-averaged** performance. That’s why.

### 3. Discrete modes + autograd = 🤯
POD gives **discrete** spatial modes. But PINNs need **continuous gradients** (thanks, backprop).

**Solution?**  
We used interpolation (splines/NNs) to make the modes differentiable.  
Hot tip: train **4 separate networks for each mode** — keeps the interpolation truer to the original CFD data.

---

## 📎 Before You Use This

This model is experimental and was made in a lab — not a control room.

> ⚠️ **Disclaimer**:  
> This model is for academic and research purposes only. It's not production-grade, and definitely not reactor-grade.  
> **Any accidents, errors, or nuclear meltdowns caused by using this model are 100% on you.**  
> You've been warned. Use wisely. Or at least don't use it to control an actual pump.

---

## 🧠 Bonus: Why this matters

CFD is great, but slow. This approach tries to bridge that with a light, learnable model that still respects physics. Think of it as:  
**“I want to *guess smart*, not *simulate slow*.”**

---

Stay curious. Stay skeptical. And always double-check what the Raynolds number(lol).
