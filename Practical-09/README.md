# Practical 9 — K-Nearest Neighbours (KNN)

## Aim
Classify wines with KNN, see why feature scaling matters, choose K without touching the test set, and draw decision regions.

## Data
`Datasets/wine.csv` — 178 wines, 13 chemical measurements, three classes (59 / 71 / 48). Stratified 75/25 split: 133 training and 45 test samples (seed 3).

## What the script does
1. Class summary of four key measurements.
2. K = 5 **without** and **with** `StandardScaler`.
3. Confusion matrix, classification report and heat-map for the scaled model.
4. K = 1…30 scored by 5-fold cross-validation on the training data; error-rate plot; the smallest odd K among the best is chosen.
5. Final model, plus the five nearest neighbours of one test wine.
6. Euclidean vs Manhattan vs Chebyshev distance.
7. Decision regions using only `flavanoids` and `color_intensity`.

## Results (from `output.txt`)
* K = 5 accuracy: **0.6667 unscaled → 0.9778 scaled** (`proline` is in the hundreds, so it dominates unscaled distances).
* Scaled K = 5 confusion matrix: one class_1 wine predicted as class_2, everything else correct.
* Lowest cross-validated error 0.037, reached for K = 17–23 and 29; chosen **K = 17**.
* Final test accuracy with K = 17: **0.9778**.
* Distance metrics: Euclidean 0.9778, Manhattan 0.9556, Chebyshev 0.9333.
* Two features alone (flavanoids + colour intensity) already give 0.9556.

With only 45 test samples, one wine changes accuracy by about 2.2 points, so small differences are not meaningful.

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn
python practical_09.py
```

## Files
`practical_09.py` · `practical_09.ipynb` · `output.txt` · `figures/` · `README.md`
