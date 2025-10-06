# Uncertainty-Aware Safe Reinforcement Learning using Control Barrier Functions

This repository contains all the material used in the Master's Thesis *“Uncertainty-Aware Safe Reinforcement Learning using Control Barrier Functions.”*  
It is designed to ensure full transparency and reproducibility of the experiments and results described in the thesis.

## Overview

The project explores a modular **Safe Reinforcement Learning** framework that combines:
- a pre-trained **Soft Actor-Critic (SAC)** policy,  
- multiple **Uncertainty Estimation (UE)** modules, and  
- a **Control Barrier Function (CBF)**-based **Safety Filter**.

The Uncertainty module outputs a scalar score representing the model’s confidence in its current state.  
If the score exceeds a threshold, the Safety Filter intervenes to minimally correct the action, ensuring safe execution while preserving task performance.

## Repository Contents

- `python/` – all Python scripts used for:
  - training of the base SAC policy and all Uncertainty Estimation models  
  - evaluation of UE + CBF configurations across environments  
  - data processing, metric aggregation, and plotting utilities  

- `NN/` – trained models used in the experiments:
  - pre-trained SAC policies  
  - trained Uncertainty Estimation networks (Probabilistic, MCD, Ensemble, RND)  
  - saved configurations and normalization statistics for direct reuse  

- `Navigation_Assets.unitypackage` – Unity package containing all assets and scenes used in the simulations:
  - procedural obstacle generation implemented through a cellular automaton algorithm  
  - custom **SideChannel** scripts for Python–Unity communication  
  - all Unity scenes for training and testing (ST, SCW, and MO)  

- `results_data/` – all tabulated experimental results:
  - DataFrames and CSVs containing aggregated metrics  
  - summary tables for each model, environment, and uncertainty threshold  
  - synthetic “score” metrics combining safety and efficiency indicators  

## Reproducibility

The repository includes everything needed to reproduce the experimental results:

1. Import the `Navigation_Assets.unitypackage` file into Unity to load all necessary scenes and assets.  
2. Use the **`NN/`** folder to access the trained SAC and Uncertainty Estimation models used in the thesis.  
3. Run the notebook **`env_test.ipynb`** to evaluate the framework directly using the provided models.  
   - The current file paths inside the notebook refer to the previous repository structure and **must be updated** to match the new organization (notably for model loading and result directories).  
4. Alternatively, use the scripts in `python/` to study or reproduce the complete training procedures for the SAC policy, UE modules, and the CBF-based Safety Filter.  
5. Aggregated metrics and plots are available in `results_data/` and `results_plots/` for direct inspection and analysis.  

## Citation

If you use this repository or reproduce its results, please cite:

> G. Aru, *Uncertainty-Aware Safe Reinforcement Learning using Control Barrier Functions*,  
> Master’s Thesis, University of Pisa, 2025.

---

*This repository provides the complete codebase, trained models, Unity package, and experimental outputs used in the thesis, ensuring open access and reproducibility of the research.*
