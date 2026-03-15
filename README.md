# AI-Based Fault Detection in Transmission Lines
### Smart Grid Fault Detection using Machine Learning

## Project Overview

This project implements an **AI-based fault detection system** for 3-phase electrical transmission lines using classical and modern machine learning techniques. The system can automatically classify five types of power system conditions — including three distinct fault types — from electrical measurements in **under 1 millisecond per sample**, making it suitable for real-time smart grid applications.

> **Domain:** Power Systems | Smart Grid | Electrical Engineering  
> **Type:** Multi-class Classification  
> **Environment:** Google Colab (Python)

---

## Objectives

- Simulate realistic 3-phase transmission line fault data with engineered electrical features
- Train and compare three ML classifiers: **Random Forest**, **SVM**, and **Neural Network**
- Evaluate models on **classification accuracy**, **F1-score**, and **detection time**
- Provide a real-time fault detection inference pipeline

---

##  Fault Types Detected

| Label | Fault Type | Description |
|-------|-----------|-------------|
| `Normal` | No Fault | Balanced 3-phase operation |
| `LG` | Line-to-Ground | One phase contacts ground |
| `LL` | Line-to-Line | Two phases short-circuit |
| `DLG` | Double Line-to-Ground | Two phases contact ground |
| `3PH` | Three-Phase Fault | All three phases short-circuit (most severe) |

---

##  ML Models Used

| Model | Key Hyperparameters |
|-------|-------------------|
| **Random Forest** | `n_estimators=200`, `max_depth=15` |
| **SVM** | `kernel=RBF`, `C=10`, `gamma=scale` |
| **Neural Network (MLP)** | `layers=[128, 64, 32]`, `activation=ReLU`, `solver=Adam` |

---

##  Features Engineered

From simulated 3-phase voltage and current measurements, **17 features** are extracted:

```
Va, Vb, Vc          → Phase voltages (pu)
Ia, Ib, Ic          → Phase currents (pu)
V_zero, I_zero      → Zero-sequence components
V_neg, I_neg        → Negative-sequence components
V_rms, I_rms        → RMS voltage and current
V_imbalance         → Voltage imbalance ratio
I_imbalance         → Current imbalance ratio
THD_V, THD_I        → Total Harmonic Distortion (%)
Power               → Three-phase active power
```

---

##  Project Structure

```
smart-grid-fault-detection/
│
├── Smart_Grid_Fault_Detection.ipynb   # Main Colab notebook (all 14 sections)
├── README.md                          # Project documentation
│
└── outputs/                           # Generated after running notebook
    ├── eda_distributions.png          # Feature distributions by fault type
    ├── correlation_heatmap.png        # Feature correlation matrix
    ├── confusion_matrices.png         # Per-model confusion matrices
    ├── feature_importance.png         # Random Forest feature ranking
    ├── model_comparison.png           # Accuracy, F1, speed comparison chart
    ├── cross_validation.png           # 5-fold CV boxplot
    └── nn_loss_curve.png              # Neural Network training loss curve
```

---

##  Getting Started

### Option 1: Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Upload `Smart_Grid_Fault_Detection.ipynb`
4. Click **Runtime → Run All**

> ✅ No setup required — all dependencies install automatically in the first cell.

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/smart-grid-fault-detection.git
cd smart-grid-fault-detection

# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Launch Jupyter
jupyter notebook Smart_Grid_Fault_Detection.ipynb
```

---

## 📊 Results

### Model Performance Summary

| Model | Accuracy | F1-Score | Detection Time |
|-------|----------|----------|----------------|
| Random Forest | ~98–99% | ~98–99% | < 0.05 ms/sample |
| SVM (RBF) | ~96–98% | ~96–98% | < 0.01 ms/sample |
| Neural Network | ~97–99% | ~97–99% | < 0.01 ms/sample |

> *Results are from simulated data; values may vary slightly each run.*

### Key Findings

- **Three-Phase faults** produce the most distinct electrical signatures, achieving near-perfect detection
- **Zero-sequence** and **negative-sequence** components are the most discriminative features
- **Random Forest** offers the best balance of accuracy and interpretability
- All models achieve **> 95% accuracy** with the engineered feature set
- Detection time **< 1 ms/sample** is well within real-time protection system requirements (typically 20–100 ms)

---

## 📈 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Libraries | Install & import all dependencies |
| 2 | Data Simulation | Generate 2000-sample, 5-class fault dataset |
| 3 | EDA | Histograms and correlation heatmap |
| 4 | Preprocessing | Scaling, encoding, train/test split (80/20) |
| 5 | Random Forest | Training, prediction, classification report |
| 6 | SVM | RBF kernel training and evaluation |
| 7 | Neural Network | MLP training with early stopping |
| 8 | Learning Curve | NN training loss visualization |
| 9 | Confusion Matrices | Side-by-side for all 3 models |
| 10 | Feature Importance | Random Forest feature ranking |
| 11 | Comparison Dashboard | Accuracy, F1, speed bar charts |
| 12 | Cross-Validation | 5-fold CV with boxplot |
| 13 | Live Detection | Real-time inference on 5 scenarios |
| 14 | Final Report | Complete summary printout |

---

##  Dependencies

```
Python        >= 3.8
scikit-learn  >= 1.0
numpy         >= 1.21
pandas        >= 1.3
matplotlib    >= 3.4
seaborn       >= 0.11
```

---

##  Future Improvements

- [ ] Integrate real dataset (e.g., IEEE 13-bus or PSCAD simulation data)
- [ ] Add deep learning models (LSTM, CNN for waveform time-series)
- [ ] Implement fault location estimation (not just classification)
- [ ] Deploy as a web dashboard using Streamlit or Gradio
- [ ] Add noise robustness testing and imbalanced class handling
- [ ] Extend to distribution network faults (underground cables)

---

##  References

- Gers, J.M. & Holmes, E.J. — *Protection of Electricity Distribution Networks*
- Anderson, P.M. — *Analysis of Faulted Power Systems*, IEEE Press
- Scikit-learn Documentation — https://scikit-learn.org
- IEEE Std C37.113 — *Guide for Protective Relay Applications to Transmission Lines*

---

## 👤 Author

**Your Name** 
Francesco De Florence
Department of Electrical & Electronic Engineering  
