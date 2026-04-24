# NYC Taxi Analytics & Machine Learning System

> An end-to-end data science pipeline built on real-world NYC taxi data, spanning scalable ETL engineering, PCA-based representation learning, heavy-tail statistical analysis, and interpretable ML modeling with GAM and XGBoost.


---

## Overview

Urban mobility data is large, noisy, and heterogeneous. This project builds an end-to-end system to extract meaningful signal from it, from raw Parquet ingestion through PCA-based representation learning to interpretable fare prediction and taxi type classification.

The system is organized into four self-contained phases, each building on the last.

---

## Pipeline at a Glance

```
Raw Taxi Data (Parquet)
        │
        ▼
┌─────────────────────────┐
│  Phase 1: Data Pipeline │  PyArrow · multiprocessing · partition optimization
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Wide Feature Table                │  (taxi_type, date, location) × hour_0…hour_23
└────────────┬───────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Phase 2: PCA + Statistical Analysis│  Representation learning · tail analysis · bootstrap
└────────────┬────────────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
┌──────────────┐  ┌────────────────────────────┐
│ Phase 3: GAM │  │ Phase 4: XGBoost Classifier │
│ Fare Predict │  │ Taxi Type Classification    │
└──────────────┘  └────────────────────────────┘
```

---

## Phases

### Phase 1 — Scalable Data Pipeline

**`01_scalable_data_pipeline/`**

Builds a production-style ETL pipeline for large-scale taxi data.

**What it does:**
- Processes Parquet data (local and S3-compatible)
- Handles schema inconsistencies across files
- Aggregates trip-level records into a wide feature table: `(taxi_type, date, pickup_location) × hour_0…hour_23`
- Filters low-signal rows (< 50 rides) and merges into a single output table

**Engineering highlights:**
- Partition-based parallel processing with `multiprocessing`
- Partition size optimization for memory efficiency
- Runtime, memory, and data-loss tracking throughout

---

### Phase 2 — PCA, Tail Analysis & Stability

**`02_pca_spatial_bootstrap_analysis/`**

Extracts and validates latent temporal patterns in the ride distribution data.

#### Representation Learning (PCA)
- Applied to hourly ride distributions
- Identifies dominant temporal patterns across the city
- Outputs variance explained and a serialized PCA model (`models/pca_model.pkl`)
- Spatial visualization of components mapped onto NYC taxi zones via Folium

#### Statistical Analysis
- Histogram and Q-Q plots of PCA coefficients
- Log-log survival plots and Hill estimator for power-law tail exponents
- **Key finding:** PCA coefficients exhibit heavy-tailed behavior, indicating bursty demand dynamics

#### Bootstrap Stability
- Resamples data to evaluate PCA robustness
- Metrics: eigenvector stability, subspace similarity, and component consistency across resamples

---

### Phase 3 — Interpretable Fare Prediction (GAM)

**`03_gam_fare_prediction/`**

A **Generalized Additive Model** for predicting taxi fares with full interpretability.

**Features used** (no fare-leakage variables):
- Trip distance and duration
- Hour of day and day of week
- Passenger count

**Model design:** Smooth spline terms for distance, duration, and time; linear terms for the rest.

**Evaluation:** RMSE, MAE, R²

**Interpretability outputs:** Partial dependence plots for distance → fare, duration → fare, and time-of-day effects.

---

### Phase 4 — Taxi Type Classification (XGBoost)

**`04_xgboost_taxi_type_classification/`**

Classifies trips as yellow or green taxi using two feature regimes.

| Model | Features | Purpose |
|-------|----------|---------|
| **Model A** | Distance, duration, time, passenger count | Raw feature baseline |
| **Model B** | PC1–PC5 from hourly ride distributions | Latent representation |

**Evaluation:** Accuracy, precision/recall/F1, confusion matrix

**Key comparison:** Raw feature space vs. PCA latent space — quantifies the value of learned representations for classification.

---

## Repository Structure

```
.
├── 01_scalable_data_pipeline/
├── 02_pca_spatial_bootstrap_analysis/
├── 03_gam_fare_prediction/
├── 04_xgboost_taxi_type_classification/
│
├── models/                  # Serialized model artifacts
├── results/
│   ├── figures/
│   └── reports/
│
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data processing | Pandas, NumPy, PyArrow, Dask |
| Machine learning | Scikit-learn, XGBoost, PyGAM / Statsmodels |
| Visualization | Matplotlib, Seaborn, Folium |
| Infrastructure | Python multiprocessing, Parquet (local + S3) |

---

## How to Run

This project is modular — each phase is independent and can be run on its own.

> ⚠️ Large datasets are not included in this repository. NYC TLC taxi data in Parquet format is required. Place data locally or update paths inside scripts/notebooks.

### 1. Setup

```bash
git clone <repo-url>
cd dsc291-2026
pip install -r requirements.txt
```

### 2. Phase 1 — Data Pipeline

```bash
cd 01_scalable_data_pipeline
python pivot_all_files.py --input-dir <path_to_parquet> --output-dir <output_path>
```

Generates the wide feature table used in all downstream phases.

### 3. Phase 2 — PCA & Analysis

```bash
cd ../02_pca_spatial_bootstrap_analysis
python pca_analysis.py
python tail_analysis.py
python bootstrap_stability.py
```

Outputs saved to `results/`.

### 4. Phase 3 — GAM Fare Prediction

```bash
cd ../03_gam_fare_prediction
jupyter notebook gam_fare_prediction.ipynb
```

### 5. Phase 4 — XGBoost Classification

```bash
cd ../04_xgboost_taxi_type_classification
jupyter notebook xgboost_classification.ipynb
```

### 6. Tests (Optional)

```bash
pytest tests/
```

---

## Key Results

| Phase | Highlight |
|-------|-----------|
| Phase 1 | Scalable pipeline processing millions of records with partition-level parallelism |
| Phase 2 | Heavy-tailed PCA coefficient distributions; stable subspace across bootstrap resamples |
| Phase 3 | Interpretable fare model with smooth partial dependence curves |
| Phase 4 | PCA features competitive with raw features for taxi type classification |

---

> This project demonstrates a complete ML workflow: **data engineering → feature representation → statistical analysis → modeling → evaluation** — with a focus on scalability, interpretability, and robustness.
