---
title: GraphACSM-Net
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🧬 GraphACSM-Net

> **A Graph Neural Network Web Server for Anticancer Small Molecule Prediction**
> 
> Predicts **anticancer activity (Active/Inactive)** and **pIC₅₀** for any molecule
> given as a SMILES string — powered by Graph Attention Networks (GAT) and a
> Multi-Scale Message-Passing (MSMP) architecture.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue?style=for-the-badge)](https://www.rdkit.org/)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation (Local)](#-installation-local)
- [Running the App](#-running-the-app)
- [Model Performance](#-model-performance)
- [Team & Contact](#-team--contact)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

**GraphACSM-Net** is a research-grade web application developed at SRMIST that uses
graph deep learning to predict the anticancer potential of small molecules against
colorectal cancer cell lines (SW480).

Given a molecule's SMILES representation, the app predicts:
- **Activity class** — Active (pIC₅₀ ≥ 7.0) or Inactive
- **pIC₅₀ value** — quantitative potency estimate
- **IC₅₀ (µM)** — derived inhibitory concentration
- **Physicochemical properties** — MW, LogP, TPSA, QED, HBD/HBA, etc.
- **2D & 3D molecular visualisation** — rendered in-browser

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔬 Single Molecule Prediction | Enter any SMILES and get results instantly |
| 📦 Batch Prediction | Upload a CSV with a `SMILES` column for bulk analysis |
| 🧪 FDA Drug Explorer | One-click pre-loaded reference anticancer drugs |
| 📊 Model Comparison | Side-by-side classification & regression metrics |
| 🌗 Dark / Light Mode | Fully themed toggle via floating button |
| 🔵 2D + 3D Viewer | RDKit SVG + 3Dmol.js interactive 3D viewer |

---

## 🏗 Architecture

```
Input SMILES
     │
     ▼
RDKit Graph Construction
  • Nodes: 21-dim atom features
  • Edges: 7-dim bond features
  • Global: 13 molecular descriptors
     │
     ▼
┌─────────────────────────────────────┐
│  Model A — SimpleGAT (Random Split) │
│  4 × GATv2Conv layers               │
│  Classification head (BCE Loss)     │
│  Regression head (MSE Loss)         │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  Model B — HighPerfMSMP (Scaffold)  │
│  4 × (GINE + GATv2 + Transformer)  │
│  GlobalAttention pooling            │
└─────────────────────────────────────┘
     │
     ▼
  Activity · pIC₅₀ · IC₅₀ (µM)
```

---

## 📁 Project Structure

```
graphacsm-net/
├── app.py
├── gat_random_aug_fold3_seed42.pt
├── msmp_scaffold_aug_fold4_seed42.pt
├── ppt_table_results.csv
├── web.png
├── requirements.txt
├── packages.txt
└── README.md
```

---

## 💻 Installation (Local)

```bash
git clone https://huggingface.co/spaces/<your-username>/graphacsm-net
cd graphacsm-net
pip install -r requirements.txt
streamlit run app.py
```

---

## ▶️ Demo Link

https://graphacsm-anticancer.streamlit.app/

---

## 📊 Model Performance

| Condition | Model | Test AUC | Test ACC | Test MCC | Test R² | Test RMSE |
|-----------|-------|----------|----------|----------|---------|-----------|
| Random + Aug | GAT | **0.9364** | **0.8703** | **0.7421** | **0.6841** | **0.6628** |
| Scaffold + Aug | MSMP | 0.8988 | 0.8213 | 0.6481 | 0.5494 | 0.7900 |

---

## 👥 Team & Contact

Developed at the **Computational Biology Lab, SRMIST**

| Name | Role |
|------|------|
| Dr. Thirumurthy Madhavan | Principal Investigator |
| Dr. A. Revathi | Co-Investigator |
| Ms. Priya Dharshini B. | Research Scholar |
| Ms. Subathra Selvam | Research Scholar |
| Priyanshu Goyal | Developer (B.Tech AI/ML) |
| Yadhisresht Harikrishnan | Developer (B.Tech AI/ML) |

📍 SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu 603203

---

## ⚠️ Disclaimer

GraphACSM-Net is a **computational research tool** for academic use only.
All predictions are *in silico* and require experimental validation.
**Not approved for medical or clinical use.**

---

*© 2026 GraphACSM-Net · SRMIST · All rights reserved*
