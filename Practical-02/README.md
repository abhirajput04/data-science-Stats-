# Practical 2 — Foundations of Statistics (Descriptive Statistics)

## Aim
Describe a dataset using measures of centre, spread and shape, compare groups, and cross-tabulate two categorical variables.

## Data
`Datasets/wine.csv` — 178 Italian wines from three cultivars (`class_0`: 59, `class_1`: 71, `class_2`: 48) with 13 chemical measurements. No missing values.

## What the script does
1. Inspects the table and decides the dataset type — **cross-sectional** (each wine measured once, no time axis).
2. Mean, median, min, max, range, standard deviation and IQR of `proline`.
3. Full descriptive table for all 13 measurements, including the coefficient of variation.
4. Histogram of `alcohol` with mean/median markers, plus skewness and kurtosis.
5. `groupby` summary per wine class using named aggregations.
6. Cross-tabulation of class against a high/low alcohol label (counts and row percentages).
7. Outlier check on `magnesium` using the 1.5 × IQR rule.

## Results (from `output.txt`)
| Statistic | proline | alcohol |
|---|---|---|
| Mean | 746.89 | 13.0006 |
| Median | 673.50 | 13.0500 |
| Std dev | 314.91 | 0.8118 |
| Min / Max | 278 / 1680 | 11.03 / 14.83 |

* Alcohol skewness is −0.05 and kurtosis −0.85, so the distribution is roughly symmetric and flatter than a normal curve.
* Mean alcohol by class: 13.745 (class_0), 12.279 (class_1), 13.154 (class_2).
* Share of "high alcohol" wines per class: 96.6 % (class_0), 9.9 % (class_1), 56.3 % (class_2).
* Four magnesium outliers were flagged (all in `class_1`); the fence is [59.5, 135.5].
* Most variable measurements by coefficient of variation: flavanoids (49 %), malic acid (48 %), colour intensity (46 %).

## How to run
```bash
pip install pandas numpy matplotlib
python practical_02.py
```
The histogram is also saved to `figures/alcohol_histogram.png`.

## Files
`practical_02.py` · `practical_02.ipynb` · `output.txt` · `figures/` · `README.md`
