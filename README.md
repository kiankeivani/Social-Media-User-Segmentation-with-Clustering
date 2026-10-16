# Social Media User Segmentation with Clustering

Unsupervised segmentation of social media users using **K-Means**, **Hierarchical (Agglomerative) Clustering**, and **DBSCAN**, based on demographic and engagement features. The project covers preprocessing, hyperparameter tuning, evaluation with internal validity metrics, cluster profiling, visualization, and business-oriented naming of the discovered segments.

> Course: Data Analysis 

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Cluster Profiles](#cluster-profiles)
- [Answers to the Project Questions](#answers-to-the-project-questions)
- [Tech Stack](#tech-stack)
- [Notes and Limitations](#notes-and-limitations)

---

## Overview

The goal is to discover distinct behavioral groups of users on a social platform and describe them in a meaningful way (e.g. "young influencers", "inactive older users"). Three clustering families are implemented and compared:

| Algorithm | Family | Hyperparameters tuned |
|---|---|---|
| K-Means | Centroid-based | `k` (Elbow + Silhouette) |
| Agglomerative (complete linkage) | Hierarchical | `k` (Silhouette) + dendrogram |
| DBSCAN | Density-based | `eps` (k-distance plot + refinement grid), `min_samples` |

## Dataset

`rawData.csv` — 200 users, 5 columns:

| Feature | Description |
|---|---|
| `UserID` | Unique user identifier (not used for clustering) |
| `Age` | User age in years |
| `FollowersCount` | Number of followers |
| `PostsCount` | Total number of published posts |
| `EngagementRate` | Average engagement rate (likes, comments, shares) |

Clustering is performed on `Age`, `FollowersCount`, `PostsCount`, and `EngagementRate`.

## Methodology

1. **Cleaning** — convert features to numeric, impute missing values with the column mean (`SimpleImputer`).
2. **Scaling** — `StandardScaler` for DBSCAN; `MinMaxScaler` + Euclidean distance matrix for hierarchical clustering.
3. **Hyperparameter tuning**
   - K-Means: Elbow plot (inertia) and Silhouette score for `k = 2…10`.
   - Hierarchical: Silhouette score for `k = 2…10` with complete linkage; dendrogram for visual inspection.
   - DBSCAN: k-distance plot with `min_samples = 5`, initial `eps` taken from the 95th percentile of k-distances, then a refinement grid (0.7×–1.3× of the guess) selecting the best Silhouette on non-noise points.
4. **Evaluation** — Silhouette, Davies–Bouldin (lower is better), Calinski–Harabasz (higher is better). For DBSCAN, metrics are computed on non-noise points only.
5. **Profiling** — descriptive statistics (count, mean, std, quartiles) per cluster for all four features.
6. **Visualization** — 2D scatter plots (Followers vs. Engagement, Engagement vs. Age, Followers vs. Age) with named clusters.

## Results

| Algorithm | Clusters | Noise | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Cluster sizes |
|---|---|---|---|---|---|---|
| K-Means (`k=3`) | 3 | – | **0.704** | **0.390** | **1334.5** | 50 / 101 / 49 |
| Hierarchical (`k=2`) | 2 | – | 0.650 | 0.499 | 639.5 | 100 / 100 |
| DBSCAN (`eps=0.436`, `min_samples=5`) | 2 | 3 | 0.647 | 0.505 | 618.2 | 99 / 98 (+3 noise) |

**Choice of `k` for K-Means.** The Silhouette score is marginally higher at `k=2`, but the Elbow plot shows a sharp inertia drop up to `k=3` and a flatter slope afterwards. `k=3` was chosen since it yields a finer, more interpretable segmentation (a distinct mid-tier group) without a meaningful loss in Silhouette.

## Cluster Profiles

### K-Means (k = 3)

| Cluster | Name | Avg. Age | Avg. Followers | Avg. Posts | Avg. Engagement | Size |
|---|---|---|---|---|---|---|
| 0 | **Young influencers** | 24.8 | 3,348 | 120.1 | 0.77 | 50 |
| 1 | **Inactive older users** | 50.7 | 365 | 26.5 | 0.14 | 101 |
| 2 | **Mid-age engaged users** | 32.1 | 1,527 | 80.2 | 0.54 | 49 |

### Hierarchical (k = 2)

| Cluster | Name | Avg. Age | Avg. Followers | Avg. Posts | Avg. Engagement | Size |
|---|---|---|---|---|---|---|
| 0 | Young influencers | 28.4 | 2,431 | 100.0 | 0.65 | 100 |
| 1 | Inactive older users | 50.9 | 360 | 26.1 | 0.14 | 100 |

### DBSCAN

| Cluster | Name | Avg. Age | Avg. Followers | Avg. Posts | Avg. Engagement | Size |
|---|---|---|---|---|---|---|
| 0 | Inactive older users | 51.0 | 358 | 26.1 | 0.14 | 99 |
| 1 | Young influencers | 28.6 | 2,423 | 99.9 | 0.65 | 98 |
| −1 | Noise / atypical users | 26.0 | 2,067 | 78.3 | 0.45 | 3 |

DBSCAN flags three atypical users (e.g. a very young, high-follower, high-posting user) as noise rather than forcing them into a cluster.

## Answers to the Project Questions

1. **Which cluster contains young users with high engagement?**
   K-Means cluster 0 (avg. age ≈ 25, engagement ≈ 0.77); the equivalent "young influencers" cluster in Hierarchical (cluster 0) and DBSCAN (cluster 1).
2. **Which cluster has the highest average followers?**
   The same "young influencers" cluster in all three methods (≈ 3,348 in K-Means, ≈ 2,431 in Hierarchical, ≈ 2,423 in DBSCAN).
3. **Can users with more than 75 posts be isolated in a single cluster?**
   82 users have `PostsCount > 75`.
   - K-Means: 50 in cluster 0 and 32 in cluster 2 → split across the two active clusters.
   - Hierarchical: all 82 in cluster 0 (the active group).
   - DBSCAN: 80 in the active cluster and 2 flagged as noise.

   Heavy posters are therefore concentrated in the active segments, but only the coarser 2-cluster solutions group them into a single cluster.


## Tech Stack

Python · NumPy · pandas · scikit-learn · SciPy · Matplotlib · Jupyter

## Notes and Limitations

- The dataset is small (200 users); results are illustrative and may not generalize.
- In the notebook, the K-Means search and fit use the raw (imputed) features, while DBSCAN uses standardized features and Hierarchical uses Min-Max scaled features. Because the features have very different scales (e.g. followers vs. engagement rate), K-Means results are dominated by `FollowersCount`; re-running K-Means on standardized data is a recommended extension.
- Silhouette-based selection favors `k=2` across methods; the final `k=3` for K-Means is a deliberate interpretability trade-off.
- The narrative analysis and cluster descriptions inside the notebook are written in Persian.
