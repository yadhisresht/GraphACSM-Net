# 🧬 GraphACSM-Net

> **A Graph Neural Network Web Server for Anticancer Small Molecule Prediction**
> 
> Predicts **anticancer activity (Active/Inactive)** and **pIC₅₀** for any molecule
> given as a SMILES string — powered by Graph Attention Networks (GAT) and a
> Multi-Scale Message-Passing (MSMP) architecture.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue?style=for-the-badge)](https://www.rdkit.org/)
[![License: Research](https://img.shields.io/badge/License-Research_Only-lightgrey?style=for-the-badge)]()

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation (Local)](#-installation-local)
- [Running the App](#-running-the-app)
- [Dataset](#-dataset)
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

## 🌐 Live Demo

> _Deployed via Streamlit Community Cloud_  
> 🔗 **[https://graphacsm-net.streamlit.app](https://graphacsm-net.streamlit.app)** ← replace with your real URL after deployment

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔬 Single Molecule Prediction | Enter any SMILES and get results instantly |
| 📦 Batch Prediction | Upload a CSV with a `SMILES` column for bulk analysis |
| 🧪 FDA Drug Explorer | One-click pre-loaded reference anticancer drugs |
| 📊 Model Comparison | Side-by-side classification & regression metrics |
| 🌗 Dark / Light Mode | Fully themed toggle via floating button |
| 📱 Mobile Responsive | Works on phones and tablets |
| 🔵 2D + 3D Viewer | RDKit SVG + 3Dmol.js interactive 3D viewer |

---

## 🏗 Architecture

```
Input SMILES
     │
     ▼
RDKit Graph Construction
  • Nodes: 21-dim atom features (atomic no., hybridisation, aromaticity, …)
  • Edges: 7-dim bond features (single/double/triple/aromatic, conjugation, …)
  • Global: 13 molecular descriptors (MW, LogP, TPSA, QED, …)
     │
     ▼
┌─────────────────────────────────────┐
│  Model A — SimpleGAT (Random Split) │  ← Recommended for general use
│  4 × GATv2Conv layers               │
│  Shared representation layer        │
│  Classification head (BCE Loss)     │
│  Regression head (MSE Loss)         │
└─────────────────────────────────────┘
     │
┌─────────────────────────────────────┐
│  Model B — HighPerfMSMP (Scaffold)  │  ← Conservative / novel scaffolds
│  4 × (GINE + GATv2 + Transformer)  │
│  Layer-weighted pooling             │
│  GlobalAttention pooling            │
└─────────────────────────────────────┘
     │
     ▼
  Activity (Active / Inactive)
  pIC₅₀  ·  IC₅₀ (µM)
```

**Data pipeline:**  
3,600 PubChem compounds → 3× SMILES augmentation → 10,800 graphs → 80/20 split (Random or Scaffold)

---

## 📁 Project Structure

```
graphacsm-net/
├── app.py                              # Main Streamlit application (all pages)
├── gat_random_aug_fold3_seed42.pt      # Pre-trained GAT model weights (Random Split)
├── msmp_scaffold_aug_fold4_seed42.pt   # Pre-trained MSMP model weights (Scaffold Split)
├── ppt_table_results.csv               # Performance metrics table (used in Results page)
├── web.png                             # Architecture / banner image
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System-level packages (for Streamlit Cloud)
└── README.md                           # This file
```

---

## 💻 Installation (Local)

### Prerequisites

- Python 3.10 or 3.11
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/graphacsm-net.git
cd graphacsm-net
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### Step 3 — Install system packages (Linux only)

```bash
sudo apt-get install -y libxrender1 libsm6 libxext6 libgl1
```

> On macOS or Windows these libraries are not needed.

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ PyTorch Geometric requires a matching PyTorch version.  
> If `pip install torch_geometric` fails, follow the
> [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## 📂 Dataset

- **Source:** PubChem BioAssay (AID 1259344 — SW480 colorectal cancer cell line)
- **Size:** 3,600 unique compounds → 10,800 after SMILES augmentation (3×)
- **Labels:**
  - Binary activity: Active if pIC₅₀ ≥ 7.0 (IC₅₀ ≤ 0.1 µM)
  - Regression target: standardised pIC₅₀
- **Split strategies:** Random split and Scaffold (Bemis–Murcko) split
- **Class balance:** 50 : 50 (Active : Inactive)

> 📥 To request the training dataset, contact the development team (see below).

---

## 📊 Model Performance

Best results (Test set) from `ppt_table_results.csv`:

| Condition | Model | Test AUC | Test ACC | Test MCC | Test R² | Test RMSE |
|-----------|-------|----------|----------|----------|---------|-----------|
| Random + Aug | GNN-2 (GAT) | **0.9364** | **0.8703** | **0.7421** | **0.6841** | **0.6628** |
| Scaffold + Aug | GNN-4 (MSMP) | 0.8988 | 0.8213 | 0.6481 | 0.5494 | 0.7900 |

> Random split GAT is the recommended model for general compound screening.  
> Scaffold split MSMP is more conservative and better suited for evaluating novel chemical series.

---

## 👥 Team & Contact

Developed at the **Computational Biology Lab, SRMIST**

| Name | Role | Department |
|------|------|------------|
| Dr. Thirumurthy Madhavan | Principal Investigator | Genetic Engineering, SoB |
| Dr. A. Revathi | Co-Investigator | Computational Intelligence, SoC |
| Ms. Priya Dharshini B. | Research Scholar | Genetic Engineering, SoB |
| Ms. Subathra Selvam | Research Scholar | Genetic Engineering, SoB |
| Priyanshu Goyal | Developer (B.Tech AI/ML) | Computational Intelligence, SoC |
| Yadhisresht Harikrishnan | Developer (B.Tech AI/ML) | Computational Intelligence, SoC |

📍 SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu 603203  
📞 +91-9944572918  
🌐 [www.srmist.edu.in](https://www.srmist.edu.in)

---

## ⚠️ Disclaimer

GraphACSM-Net is a **computational research tool** intended for academic and
research purposes only. All predictions are *in silico* and require independent
experimental validation before any clinical or pharmaceutical application.
This tool is **not approved for medical or clinical use**.

---

*© 2026 GraphACSM-Net · SRMIST · All rights reserved*
