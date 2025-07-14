# 🌀 POD-PINN Based Main Pump Flow Field Predictor

## What's this thing even for?

This repo is for predicting the flow field of a **lead-cooled fast reactor main pump** under different operating conditions — quickly. Like, way faster than rerunning CFD every single time. The idea is: get the trends, save your time, and maybe (just maybe) shorten the design loop in the early stages.

## Compared against what?

The model was trained and tested using good ol’ CFD results — **Ansys CFX** simulations, to be exact. It's not just guessing in the dark.

## So... how well did it do?

Okay, we’ll be honest — not perfect. We’re dealing with **very high Reynolds numbers**, and we only used **4 POD modes**. That captured about 95% of the energy, but turns out, 95% isn't enough when the flow gets wild. (Even in some more linears occasions e.g. Only Thermal Conduction Evolved)

Why the error?  
- High nonlinearity, few modes = still a trade-off between speed and accuracy (error < 40% in the paper)
- Reynolds numbers are huge → behavior becomes chaotic  
- The method’s better at catching **flow trends** (correlation > 0.7 in the paper), but struggles with absolute precision

In short: **great for trend detection, not so great for exact numbers**. But hey, sometimes that's all you need.

## Wait, the code doesn't match the paper?

Yep. In fact, this repository is an improved research version — with brand-new data, better-performing programs, and, of course, **more sense of humour**. The original paper was based on earlier results — and, well, **conference page limits are brutal**, plus **some of the data is confidential**. So, a lot of juicy details had to be left on the cutting room floor. What you see here is the “director’s cut” — a more refined, fleshed-out version.

Funny story: I didn’t even realize neural network results could be kept consistent and reproducible using random seed control until the first peer review of another one of our projects. Before that, I was a complete noob in CS & AI and had no clue what a random seed even meant. That little revelation is actually how the DRQI repo was born. I am still trying to get involved because I constantly think being an CS/AI researcher is cool, but before that I must manage to improve my functional analysis skill, to dive deeper into the model structure.

---

## How does it work?

We combined two fancy techniques:

### 1. POD + PINN = ❤️
- **POD** (Proper Orthogonal Decomposition) breaks down the CFD flow fields into modes.
- Then **PINNs** (Physics-Informed Neural Networks) try to regress the modal coefficients under different working conditions.

**Equations?**
- POD:  

$$
\mathbf{U}(\mathbf{x}, t) \approx \sum_{i=1}^{N} a_i(t) \phi_i(\mathbf{x})
$$

- PINN:  
  Neural networks trained with physical constraints (governing equations baked into the loss function).

### 2. Radial layering (a.k.a. how we avoided running out of RAM)
- Initially, we ran POD **layer by layer** along the radial direction (cylindrical slices).
- But this caused issues when trying to interpolate the modal coefficients in the **r** direction.
- So the paper only reported **layer-averaged** performance. That’s why.

### 3. Discrete modes + autograd = 🤯
POD gives **discrete** spatial modes. But PINNs need **continuous gradients** (thanks, backprop).

### 4. Graphical Abstract
![Graphical Abstract](graphical_abstract.png)

📌For academic presentation only

**Solution?**  
We used interpolation (3-nearest neighbor in the paper, not so accurate as we expected when computing $\frac{\partial u}{\partial r}$ due to the r-zone is relatively narrow/NNs, we used now, aslo not accurate enough...) to make the modes differentiable.  
Hot tip: train **4 separate networks for each mode** — keeps the interpolation truer to the original CFD data.
tip: 

---

## 📎 Before You Use This

This model is experimental and was made in a lab — not a control room.

> ⚠️ **Disclaimer**:  
> This model is for academic and research purposes only. It's not production-grade, and definitely not reactor-grade.  
> **Any accidents, errors, or nuclear meltdowns caused by using this model are 100% on you.**  
> You've been warned. Use wisely. Or at least don't use it to control an actual LFR MCP.

---

## 🧠 Bonus: Why this matters

CFD is great, but slow. This approach tries to bridge that with a light, learnable model that still respects physics (see the explanation of the results in the paper). Think of it as:  
**“I want to *guess smart*, not *simulate slow*.”**

During my undergrad, my thesis was all about building a digital twin for the LFR MCP. Back then, I went with a **pure PINN** approach. It worked — kind of. The relative error in the flow field was around **15%**, which isn’t bad.  
**BUT** the dataset was *huge*, training was *painfully slow*, and I nearly didn’t graduate because of it 💀

So I’ve been on the hunt for a better way ever since. This POD-PINN thing?  
It’s a work-in-progress. It **needs way less data** (yay!) but the accuracy? Still not ideal (meh).  
Call it an *unsuccessful attempt at being efficient*, but hey — at least it's a step.

Stay curious. Stay skeptical. And always double-check what the Raynolds number(lol). Enjoy the code in `model_summon` filefolder.
