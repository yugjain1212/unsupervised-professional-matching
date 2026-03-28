# Professional Profile Clustering & Compatibility Engine

> 4th Semester Machine Learning Project — Unsupervised Learning Pipeline  
> Course: Machine Learning | Semester 4

---

## Overview

This project builds an unsupervised machine learning pipeline that identifies meaningful similarities between professional profiles and surfaces relevant connections — without using any labeled outputs, compatibility scores, or profile identifiers.

The system operates entirely on profile-level attributes to discover natural groupings and enable interpretable professional recommendations.

---

## Problem Statement

Professional networking platforms generate millions of connections, but the majority provide little mutual value. The goal of this project is to learn latent structure in professional profiles using unsupervised methods — grouping similar users, and within each group, ranking profiles by cosine similarity to surface high-value connections.

No ground truth labels are used at any stage of modelling. The `compatibility_pairs.csv` file provided with the dataset is explicitly excluded from the pipeline to preserve the unsupervised nature of the system.

---

## Dataset

**Source:** [LinkedIn Compatibility Dataset — 50K Profiles](https://www.kaggle.com/datasets/likithagedipudi/linkedin-compatibility-dataset-50k-profiles)

**File used:** `profiles.csv` (50,000 rows)

| Column | Type | Usage |
|---|---|---|
| `profile_id` | identifier | Mapped to anonymous index, excluded from model |
| `name`, `email` | PII | Dropped before modelling |
| `current_role` | text | Bucketed into 12 functional groups → one-hot encoded |
| `current_company` | text | Dropped (too high cardinality, no generalizable signal) |
| `industry` | categorical | One-hot encoded (rare industries grouped as "Other") |
| `years_experience` | numeric | StandardScaler |
| `seniority_level` | ordinal | Encoded: entry=0, mid=1, senior=2, executive=3 |
| `skills` | JSON array | Parsed → multi-hot encoded (top 100 skills) |
| `experience` | JSON array | Parsed → `num_roles` + `avg_tenure` (engineered features) |
| `education` | JSON array | Parsed → `highest_degree` ordinal (0=none … 3=PhD) |
| `connections` | numeric | log1p transform → StandardScaler |
| `goals`, `needs`, `can_offer` | JSON / text | Concatenated → TF-IDF (top 50 terms) |
| `location` | text | Country/region extracted → one-hot encoded (weak feature) |

> `compatibility_pairs.csv` is **never loaded** in this pipeline.

---

## Project Structure

```
professional-compatibility-engine/
│
├── data/
│   ├── raw/                        # Original profiles.csv (not committed)
│   ├── processed/                  # X_processed.npy — saved feature matrix
│   └── mappings/                   # anon_id_map.csv — profile_id to row index
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All constants: column names, paths, hyperparams
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── drop_columns.py         # PII + label leakage removal
│   │   ├── text_pipeline.py        # TF-IDF for skills, goals, needs, can_offer
│   │   ├── categorical_pipeline.py # One-hot encoding for industry, role, location
│   │   ├── numerical_pipeline.py   # Scaling + imputation for numeric columns
│   │   └── id_mapper.py            # profile_id ↔ anonymous row index mapping
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_union.py        # Combines all pipelines into one matrix
│   │   └── dimensionality.py       # PCA for reducing feature space
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py           # K-Means baseline + evaluation
│   │   ├── compatibility.py        # Within-cluster cosine similarity + display
│   │   └── embedding.py            # Reserved for Week 16 comparison
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── intrinsic_metrics.py    # Silhouette, Davies-Bouldin, Calinski-Harabasz
│   │   └── visualization.py        # t-SNE plots, cluster heatmaps, elbow curves
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Pipeline validation + feature matrix assembly
│   ├── 03_clustering.ipynb         # K-Means, PCA, elbow, silhouette, recommender
│   └── 04_results.ipynb            # Week 16 — model comparison + final analysis
│
├── outputs/
│   ├── models/                     # Saved sklearn pipelines (.pkl)
│   ├── clusters/                   # Cluster assignment CSVs (anonymous IDs only)
│   └── plots/                      # All EDA and model visualisation figures
│
├── requirements.txt
├── .gitignore
├── README.md
└── main.py                         # End-to-end pipeline runner (Week 16)
```

---

## ML Pipeline

```
profiles.csv (50,000 rows)
        ↓
[1] Drop PII + leakage columns (name, email, compatibility score)
        ↓
[2] Map profile_id → anonymous row index
        ↓
[3a] TF-IDF       → skills, goals, needs, can_offer
[3b] One-hot      → industry, role bucket, location
[3c] Ordinal      → seniority_level
[3d] Scale        → years_experience, connections (log1p first)
[3e] Engineer     → num_roles, avg_tenure from experience JSON
        ↓
[4] hstack all blocks → sparse feature matrix X
        ↓
[5] PCA → retain 90–95% variance → X_pca
        ↓
[6] K-Means (K selected via elbow + silhouette score)
        ↓
[7] Within-cluster cosine similarity → ranked profile recommendations
        ↓
[8] Display: full profile details fetched from df_original by row index
```

---

## Running the Notebooks

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/professional-compatibility-engine.git
cd professional-compatibility-engine
pip install -r requirements.txt
```

Download `profiles.csv` from [Kaggle](https://www.kaggle.com/datasets/likithagedipudi/linkedin-compatibility-dataset-50k-profiles) and place it at:

```
data/raw/profiles.csv
```

Then run notebooks in order:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_clustering.ipynb
```

---

## Key Design Decisions

**Why unsupervised?** The dataset includes a `compatibility_score` label. This project deliberately excludes it to demonstrate that meaningful professional similarity can be discovered from profile structure alone — without any human-annotated ground truth.

**Why PCA before K-Means?** The raw feature matrix has hundreds of dimensions due to multi-hot skill encoding and TF-IDF. K-Means degrades significantly in high dimensions (curse of dimensionality). PCA compresses the matrix while retaining 90–95% of variance, making cluster boundaries meaningful.

**Why cosine similarity for recommendations?** After clustering (coarse grouping), cosine similarity ranks profiles within a cluster by directional similarity of their feature vectors — making it robust to differences in vector magnitude, which is important for sparse multi-hot skill vectors.

**Why not use `compatibility_pairs.csv`?** Using pairwise compatibility scores would convert this into a supervised or semi-supervised problem. The core research question is whether latent structure in profile attributes alone is sufficient for meaningful grouping.

---

## Evaluation Metrics

Since no ground truth labels exist, evaluation is fully intrinsic:

| Metric | What it measures |
|---|---|
| Silhouette Score | How well-separated and cohesive clusters are (-1 to 1, higher is better) |
| Davies-Bouldin Index | Average similarity between clusters (lower is better) |
| Calinski-Harabasz Index | Ratio of between-cluster to within-cluster dispersion (higher is better) |

---

## Results (Week 11 — Baseline)

> To be updated after notebook 03 is complete.

- Optimal K: TBD
- Silhouette Score: TBD
- Davies-Bouldin Index: TBD
- Cluster interpretations: TBD

---

## Roadmap

| Week | Milestone | Status |
|---|---|---|
| 5 | Project proposal + dataset selection | Done |
| 11 | EDA + preprocessing + K-Means baseline | In progress |
| 16 | Model comparison + poster + presentation | Pending |

---

## Authors

> Add your names and roll numbers here.

---

## License

This project is for academic purposes only. Dataset sourced from Kaggle under its original terms of use.