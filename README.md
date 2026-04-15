# Architectural Style Classifier

CNN-based web app for classifying architectural styles from building photos — comparing four deep learning architectures across 24 style categories.

**[→ Live Demo](https://huggingface.co/spaces/aksrvl/architectural-style-classifier)** · **[→ Research](https://aksrvl.github.io/architectural-style-classifier)**

---

## Overview

Architectural style classification has its challenges. Style boundaries are blurry — styles evolved gradually and borrowed elements from each other. The intra-class variability is high in some styles - from small living homes to large governmental buildings. Street photos add noise, shadows and occlusions.

This project treats it as a 24-class image classification problem using transfer learning on the [Architectural Styles Dataset](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset) (9,761 images). Each model was trained three times and results are averaged for stability.

---

## Models & Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **ConvNeXt-Tiny** | **78.09%** | **77.95%** |
| ResNet50 | 75.15% | 75.35% |
| ResNet18 | 73.04% | 72.95% |
| EfficientNet-B0 | 72.63% | 72.64% |

All models trained with transfer learning on ImageNet weights, data augmentation, L2 regularization, and early stopping.

---

## Key Findings

- **Styles with distinctive visual features** (Achaemenid, Ancient Egyptian, Gothic) achieve F1 above 0.86 — minimal overlap with other classes makes them easier to classify
- **High intra-class variability** is the main challenge — Edwardian, Bauhaus, and Beaux-Arts range from small residential buildings to large civic structures
- **Historically related styles confuse the model** — biggest confusion between International and Bauhaus (19 errors) and Palladian and Georgian (15 errors)
- **ConvNeXt-Tiny outperforms ResNet50 by ~3%** despite being a newer architecture with different scaling strategy — suggesting hyperparameter choices matter per architecture

---

## Stack

Python · PyTorch · torchvision · FastAPI · Docker · Hugging Face Spaces · scikit-learn

---

## Background

Bachelor's thesis project at Kyiv-Mohyla Academy (NaUKMA), Software Engineering, 2026.
Supervisor: Oletskyy Oleksiy Vitaliyovych.
