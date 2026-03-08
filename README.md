# End-to-End Machine Learning Pipeline

A complete, production-style machine learning pipeline covering every stage from raw data ingestion to model deployment — built to demonstrate real-world ML engineering practices beyond just notebook experimentation.

---

## Problem Statement

Most ML tutorials end at model training. This project goes further: starting from raw data, through exploratory analysis, feature engineering, model selection, evaluation, threshold optimisation, and final deployment as an interactive scoring interface.

The use case is **customer churn prediction** for a telecom dataset — a high-value business problem where the cost of false negatives (missing churners) and false positives (over-targeting loyal customers) must be explicitly balanced.

---

## Pipeline Overview

```
Raw Data (CSV)
      │
      ▼
01. Exploratory Data Analysis
    (distributions, missing values, class imbalance, correlations)
      │
      ▼
02. Data Cleaning and Preprocessing
    (nulls, duplicates, encoding, scaling)
      │
      ▼
03. Feature Engineering
    (interaction features, tenure bands, service bundles)
      │
      ▼
04. Model Training and Comparison
    (Logistic Regression, Random Forest, Gradient Boosting)
      │
      ▼
05. Evaluation
    (ROC-AUC, precision-recall curve, confusion matrix)
      │
      ▼
06. Threshold Optimisation
    (business-aligned operating point for retention targeting)
      │
      ▼
07. Deployment
    (Streamlit scoring interface with risk band output)
```

---

## Key Results

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic Regression (baseline) | — | See notebook |
| Random Forest | — | See notebook |
| Gradient Boosting (final) | — | Selected for deployment |

Threshold selected to optimise recall for high-risk customers while maintaining acceptable precision for targeted retention campaigns.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

- Python, pandas, NumPy
- scikit-learn (model training, evaluation, preprocessing)
- matplotlib, seaborn (visualisation)
- Streamlit (deployment interface)
- Jupyter Notebook

---

## Project Structure

```
End-to-End-ML-Pipeline/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_preprocessing.ipynb        # Cleaning and feature engineering
│   ├── 03_modelling.ipynb            # Training, evaluation, comparison
│   └── 04_threshold_analysis.ipynb   # Operating threshold selection
│
├── app/
│   └── streamlit_app.py              # Scoring interface
│
├── models/
│   └── churn_model.pkl               # Serialised trained model
│
├── data/
│   └── telecom_churn.csv             # Dataset (source: Kaggle)
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/zainhammagi12/End-to-End-Machine-Learning-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/streamlit_app.py

# Or explore the notebooks in order
jupyter notebook notebooks/
```

---

## Business Context

The model outputs a **churn probability score** and assigns each customer to a risk band:

| Risk Band | Probability | Recommended Action |
|-----------|-------------|-------------------|
| High | > 0.70 | Priority outreach, retention offer |
| Medium | 0.40 – 0.70 | Monitoring, light-touch engagement |
| Low | < 0.40 | Standard service |

The threshold between High and Medium was tuned to maximise recall for the High band — ensuring the most at-risk customers are not missed — while keeping the High band small enough to be actionable for a retention team.

---

## Author

**Zain Hammagi** — [linkedin.com/in/zain-hammagi](https://linkedin.com/in/zain-hammagi) · [zainhammagi.github.io](https://zainhammagi.github.io)
