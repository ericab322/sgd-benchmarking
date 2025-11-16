# Theoretically-Grounded SGD Framework

This repository contains a reproducible experimental framework for studying stochastic gradient descent (SGD) on convex and non-convex objectives.
It was developed as an independent research project to explore how empirical behavior compares with classical convergence theory.

## Overview

The framework provides end-to-end tools for:
  - Polynomial and linear regression (convex objectives)
  - Two-layer neural networks (non-convex objectives)
  - Stepsize strategies: fixed, diminishing, and halving
  - Automated estimation of theoretical constants: Lipschitz smoothness, strong convexity, and gradient-moment bounds
It supports both synthetic and CSV-based real datasets and logs full experiment metadata for reproducibility.
## Key Features
- Modular SGD framework for convex and nonconvex optimization
- Built-in tracking of convergence metrics (objective history, gradient norms, distance to optimum)
- Reproducible experiment logging with per-seed CSV outputs
- Configurable model complexity (polynomial degree, network width)
- Ready-to-run experimental modes:
    - convex_model
    - convex_sample
    - nonconvex_model
    - nonconvex_sample
---

## Setting Up the Environment
This project requires specific dependencies to ensure compatibility and reproducibility. These dependencies are listed in the environment.yaml file. Follow the steps below to set up the environment:

1. **Locate the `environment.yaml` file**  
   The YAML file specifies the environment name and the required dependencies for this project.

2. **Create the environment using Conda**  
   Run the following command to create the environment:

   ```bash
   conda env create -f environment.yaml
3. **Activate the Environment**
    Once the environment is created, activate it using the following command:

    ```bash
    conda activate sgd-regression

This repository is intended as an **educational and research resource** to bridge the gap between theoretical SGD guarantees and observed empirical behavior.
