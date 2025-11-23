# Adversarial Attacks Comparison

Experimental comparison of FGSM, PGD, and DeepFool ,  analyzing trade-offs between performance, perceptibility, and transferability



##  Overview

This project implements and compares three classical adversarial attack methods — **FGSM**, **PGD**, and **DeepFool** — using **PyTorch**.  
Experiments are conducted on the **MNIST** and **CIFAR-10** datasets.

Our goal is to provide a **clear, reproducible, and interpretable** experimental study of how different attack strategies impact model robustness, visual perturbation quality, and computational cost.

Deliverables include:
- Clean, modular, and reproducible code
- Well-documented Jupyter notebooks
- Visual and quantitative results (accuracy vs ε, perturbation norms, transferability matrix)
- A concise **technical report** and **presentation slides**

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Project Objectives](#project-objectives)
3. [Implementation Plan (A–Z)](#implementation-plan-a–z)
4. [Expected Results & Analysis](#expected-results--analysis)



## Background & Motivation

Deep neural networks are vulnerable to small, human-imperceptible perturbations known as **adversarial examples** — tiny pixel changes that can cause large misclassifications.  
These vulnerabilities pose critical challenges for **security, reliability, and interpretability** in ML systems.

Rather than training a noise generator (as other teams might do), our project performs a **systematic and reproducible comparison** of standard white-box attacks to highlight practical **trade-offs** between:

- **Attack strength:** impact on model accuracy  
- **Perturbation visibility:** measured with L∞ and L2 norms  
- **Computation cost:** time per image / per batch  
- **Transferability:** how well adversarial samples fool other architectures  

This focus on reproducibility and interpretability makes the project useful for both **academic study** and **real-world ML reliability** analysis.



##  Project Objectives

### Core Goals
- **Implement** three well-known adversarial attacks using PyTorch:
  - `FGSM` (Fast Gradient Sign Method)
  - `PGD` (Projected Gradient Descent)
  - `DeepFool`
- **Evaluate** each attack in terms of:
  - Model accuracy (clean vs attacked)
  - Perturbation norms (L2)
  - Average generation time per image
  - Transferability between architectures
- **Test** on:
  - **MNIST** → lightweight, visual debugging
  - **CIFAR-10** → realistic, more complex setup
- **Deliver** professional and reproducible materials:
  - Modular code, organized notebooks, plots, report, and slides

### Optional Extensions
- Implement **Adversarial Training** (PGD-based)  
- Study how including adversarial samples during training affects robustness and cost.


## What We Learned : 

Throughout this project, we implemented and analyzed three major adversarial attacks : FGSM, PGD, and DeepFool on both MNIST and CIFAR-10. This allowed us to study how different attacks impact model predictions in terms of:

- Attack strength (accuracy drop under perturbations)

- Perturbation magnitude (L2 distances)

- Visual perceptibility of adversarial noise

- Transferability across datasets and models

A key finding of our experiments is the extreme vulnerability of standard neural networks:
even tiny, human-imperceptible perturbations can drop accuracy from 99% to 0% under strong iterative attacks such as PGD.

To address this limitation, we implemented PGD adversarial training, a defense strategy where adversarial examples are generated during training so the model learns to resist them. This method, popularized by Madry et al. (2018), is considered the gold standard in adversarial robustness research.

Our results clearly demonstrate the benefits:

A small reduction in clean accuracy (98.7% → 97.1%)

A huge improvement in robustness, with PGD accuracy rising from 0% (normal model) to ≈ 79% (robust model)

This confirms that adversarial training greatly enhances model stability, making it significantly more resistant to gradient-based attacks.

Overall, this project helped us understand:

- How adversarial attacks exploit model gradients

- Why iterative attacks (PGD, DeepFool) are stronger than single-step attacks (FGSM)

- How robustness can be measured using accuracy curves, norms, and visualization

- Why adversarial defenses are essential for deploying ML models in safety-critical environments

### Environment Setup
- Create a Python environment (`conda` or `venv`)  
- Dependencies:
  ```bash
  torch
  torchvision
  numpy
  pandas
  matplotlib
  tqdm
  scikit-image
