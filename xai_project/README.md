# PhishShield — AI-Powered Phishing Detection System
## PFE 2026 | Emna Ismail

---

## Project Overview

PhishShield is a complete phishing URL detection system that combines:

- **Machine Learning** (XGBoost, LightGBM, Random Forest) — trained on 101,063 URLs
- **Deep Learning** (CNN, LSTM, MLP) — same dataset, for comparative study
- **Explainable AI / XAI** — SHAP, LIME, and PFI (Triangular Validation)
- **Adversarial Robustness** — FGSM and PGD attack simulation
- **Statistical Validation** — Bootstrap CI (95%) and ANOVA for all models
- **Web Application** — FastAPI backend + Next.js frontend

---

## How to Run the Project (Step by Step)

### Prerequisites

```bash
# Python 3.10+ required
pip install -r requirements.txt
pip install tensorflow    # For Deep Learning models
```

### Step 1 — Train the ML Models and Run Full XAI Pipeline

```bash
cd xai_project
python main.py
```

This script:
- Trains XGBoost, LightGBM, and Random Forest on the PhishFOE dataset
- Computes SHAP, LIME, and PFI explanations (Triangular XAI Validation)
- Evaluates adversarial robustness (FGSM + PGD attacks)
- Generates all reports and visualizations in `outputs/`

---

### Step 2 — Display Full Metrics for ML Models (Answers Supervisor Feedback)

```bash
python show_full_metrics.py
```

Computes and displays:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC for all 3 ML models
- Confusion matrices with False Positive / False Negative counts
- Saves reports to `outputs/validation/full_metrics_report.md`

**Why this matters**: Accuracy alone is insufficient for cybersecurity tasks.
False Negatives (missed phishing attacks) are especially dangerous.

---

### Step 3 — Train Deep Learning Models (CNN, LSTM, MLP)

```bash
python train_deep_learning.py
```

This script:
- Trains MLP, CNN, and LSTM on the same PhishFOE dataset
- Uses EarlyStopping and ReduceLROnPlateau to avoid overfitting
- Saves trained models to `outputs/models/deep_learning/`
- Computes all 5 metrics for each DL model
- Generates a ML vs DL comparison table

**Expected runtime**: ~30–50 minutes (dataset has 101,063 samples)

---

### Step 4 — Statistical Validation for Deep Learning (Bootstrap + ANOVA)

```bash
python validate_deep_learning.py
```

Applies the same rigorous validation methodology used for ML models:

1. **Bootstrap Confidence Intervals** (200 resamples, 95% CI)
   - Proves model stability — not a "lucky" result
   - Reference: Efron & Tibshirani (1993)

2. **ANOVA** (One-Way Analysis of Variance)
   - Tests whether performance differences between models are statistically significant
   - Reference: Fisher (1921)

3. **Complete ML vs DL Comparison Table**
   - All 6 models side by side
   - All 5 metrics + FP/FN counts

Outputs saved to:
- `outputs/validation/dl_bootstrap_anova_report.md`
- `outputs/validation/ml_vs_dl_full_comparison.md`
- `outputs/validation/ml_vs_dl_metrics.json`
- `outputs/visualizations/dl_bootstrap_ci.png`
- `outputs/visualizations/ml_vs_dl_comparison.png`

---

### Step 5 — Adversarial Robustness for Deep Learning (FGSM + PGD)

```bash
python robustness_deep_learning.py
```

This script:
- Applies FGSM (Fast Gradient Sign Method, Goodfellow et al. 2014) to MLP, CNN, LSTM
- Applies PGD (Projected Gradient Descent, Madry et al. 2018) — the strongest attack
- Computes accuracy drop under each attack for all 3 DL models
- Generates ML vs DL robustness comparison chart
- **Generates the FINAL comprehensive report** (`FINAL_ML_VS_DL_REPORT.md`)
  covering: performance metrics + Bootstrap CI + ANOVA + robustness for all 6 models

Outputs saved to:
- `outputs/robustness/dl_robustness_metrics.json`
- `outputs/robustness/visualizations/dl_robustness_comparison.png`
- `outputs/robustness/visualizations/ml_vs_dl_robustness.png`
- `outputs/validation/FINAL_ML_VS_DL_REPORT.md` ← **Main report for professor**

---

### Step 6 — Statistical Validation for ML Models (Bootstrap + ANOVA)

```bash
python main_validation.py
```

Same Bootstrap CI and ANOVA for XGBoost, LightGBM, and Random Forest.
Outputs saved to `outputs/validation/`.

---

### Step 6 — Launch the Web Application

#### Terminal 1 — Start the FastAPI Backend

```bash
cd xai_project
uvicorn api.main:app --reload --port 8000
```

The API will be available at: http://localhost:8000
Interactive API documentation: http://localhost:8000/docs

#### Terminal 2 — Start the Next.js Frontend

```bash
cd phishshield-frontend
npm install   # First time only
npm run dev
```

The web application will be available at: http://localhost:3000

---

## Project Structure

```
TRAVAIL PHISHING 2026/
│
├── xai_project/                    ← Main Python project
│   ├── main.py                     ← Step 1: Full ML + XAI + Robustness pipeline
│   ├── show_full_metrics.py        ← Step 2: Display all 5 metrics for ML models
│   ├── train_deep_learning.py      ← Step 3: Train CNN, LSTM, MLP
│   ├── validate_deep_learning.py   ← Step 4: Bootstrap + ANOVA for DL models
│   ├── main_validation.py          ← Step 5: Bootstrap + ANOVA for ML models
│   │
│   ├── api/                        ← FastAPI REST API
│   │   ├── main.py                 ← API entry point
│   │   └── routes.py               ← Prediction endpoints
│   │
│   ├── xai/                        ← Explainable AI module
│   │   ├── shap_explainer.py       ← SHAP (game theory-based explanations)
│   │   ├── lime_explainer.py       ← LIME (local surrogate explanations)
│   │   ├── pfi_explainer.py        ← PFI (permutation feature importance)
│   │   └── xai_validation.py      ← Triangular cross-validation (SHAP+LIME+PFI)
│   │
│   ├── robustness/                 ← Adversarial robustness module
│   │   └── robustness_eval.py      ← FGSM and PGD attack simulation
│   │
│   ├── validation/                 ← Statistical validation module
│   │   ├── bootstrap_ci.py         ← Bootstrap Confidence Intervals
│   │   └── anova_test.py           ← ANOVA statistical test
│   │
│   ├── utils/                      ← Shared utilities
│   │   ├── data_loader.py          ← Dataset loading and preprocessing
│   │   ├── model_trainer.py        ← Model training and evaluation (5 metrics)
│   │   └── visualizer.py           ← Plotting utilities
│   │
│   └── outputs/                    ← All generated results
│       ├── models/                 ← Saved ML models (.pkl, .joblib)
│       │   └── deep_learning/      ← Saved DL models (.keras)
│       ├── validation/             ← Reports (Bootstrap, ANOVA, Metrics)
│       ├── visualizations/         ← PNG charts and figures
│       └── explanations/           ← XAI reports (Markdown)
│
├── phishshield-frontend/           ← Next.js web application
│   ├── app/
│   │   ├── page.tsx                ← Home page (URL input)
│   │   ├── analyze/page.tsx        ← Analysis results + metrics table
│   │   └── robustness/page.tsx     ← Robustness testing page
│   └── lib/api.ts                  ← API client (connects to FastAPI)
│
└── Codes/
    └── PhishFOE_ds/
        └── PhishFOE_ds.csv         ← Main dataset (101,063 URLs, 29 features)
```

---

## Evaluation Metrics — Why 5 Metrics, Not Just Accuracy

| Metric | Formula | Why It Matters in Cybersecurity |
|--------|---------|--------------------------------|
| **Accuracy** | (TP+TN) / Total | General correctness — can be misleading on imbalanced data |
| **Precision** | TP / (TP+FP) | Low precision = many legitimate URLs blocked (user frustration) |
| **Recall** | TP / (TP+FN) | **Low recall = phishing attacks escape detection → CRITICAL RISK** |
| **F1-Score** | 2·P·R / (P+R) | Best single metric when both FP and FN matter (imbalanced datasets) |
| **AUC-ROC** | Area under ROC curve | Model's ability to rank phishing higher than legitimate URLs |

> **Supervisor's feedback addressed**: The system now reports all 5 metrics
> for all 6 models (3 ML + 3 DL), including False Positive and False Negative counts.

---

## Statistical Validation — Why Bootstrap and ANOVA?

### Bootstrap Confidence Intervals
- **Purpose**: Prove that the reported accuracy is stable, not a statistical fluke
- **Method**: Re-evaluate the model 200 times on random subsamples of the test set
- **Result**: A 95% CI of [98.80%, 98.92%] means we are 95% confident the true accuracy falls in this range
- **Reference**: Efron & Tibshirani (1993) — used in ~70% of ML research papers

### ANOVA
- **Purpose**: Prove that differences between models are statistically significant
- **Method**: Compare the variance *between* models to variance *within* each model's bootstrap distribution
- **Result**: If p < 0.05, we conclude the best model is truly better — not just lucky
- **Reference**: Fisher (1921) — foundational statistical test in research

---

## Key Results Summary

### Machine Learning Models (PhishFOE Dataset, 101,063 URLs)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FP | FN |
|-------|----------|-----------|--------|----------|---------|-----|-----|
| ⭐ **LightGBM** | **98.91%** | 99.05% | **98.72%** | **98.88%** | **99.89%** | 94 | **127** |
| XGBoost | 98.85% | 99.05% | 98.61% | 98.83% | 99.88% | 94 | 138 |
| Random Forest | 98.83% | **99.16%** | 98.44% | 98.80% | 99.89% | **83** | 154 |

### Why LightGBM is the Best Choice
- Highest F1-Score (98.88%) → best balance of Precision and Recall
- Fewest False Negatives (127) → catches the most phishing attacks
- Fastest inference (<2ms per URL) → suitable for real-time API deployment

---

## Contact

**Student**: Emna Ismail
**Year**: 2026
**Project**: Phishing Detection with Explainable AI and Adversarial Robustness
