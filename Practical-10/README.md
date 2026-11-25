# Practical 10 — Principal Component Analysis (PCA)

## Aim
Explore a 30-dimensional dataset, standardise it, and reduce it to two principal components while keeping as much variance as possible.

## Data
`Datasets/breast_cancer.csv` — 569 samples (357 benign, 212 malignant) with 30 measurements: the mean, standard error and "worst" value of ten properties of cell nuclei.

## What the script does
1. `describe()`-style summary; the largest column mean is about 232,000× the smallest, which shows why scaling is needed.
2. Box plots of radius, texture and concavity by diagnosis.
3. Scatter plot of radius against texture.
4. Covariance matrix of the ten "mean" measurements (heat-map on a signed log scale, because `mean area` has variance 123,844 while `mean fractal dimension` has about 0.00005).
5. `StandardScaler` (mean 0, standard deviation 1).
6. PCA **by hand**: covariance of the scaled data → `np.linalg.eigh` → sort → project.
7. PCA with scikit-learn, then a check that both methods agree (up to sign).
8. Cumulative variance, scree chart, and the features that load most on PC1.

## Results (from `output.txt`)
| Component | Explained variance |
|---|---|
| PC1 | 44.27 % |
| PC2 | 18.97 % |
| **Both** | **63.24 %** |

* The manual eigen-decomposition matches scikit-learn: `True`.
* Components needed: 80 % → 5, 90 % → 7, **95 % → 10** (down from 30).
* Strongest PC1 loadings: mean concave points (0.261), mean concavity (0.258), worst concave points (0.251), mean compactness (0.239), worst perimeter (0.237).
* In the 2-D plot, benign and malignant samples separate mainly along PC1, with some overlap in the middle.

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn
python practical_10.py
```
(Box plots use `tick_labels`, so matplotlib 3.9 or newer is needed.)

## Files
`practical_10.py` · `practical_10.ipynb` · `output.txt` · `figures/` · `README.md`
