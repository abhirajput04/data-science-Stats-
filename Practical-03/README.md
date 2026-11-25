# Practical 3 — Exploratory Data Analysis

## Aim
Explore an unfamiliar table step by step: structure, summary statistics, derived categories, group comparisons, cross-tabulation and basic charts.

## Data
`Datasets/diabetes_progression.csv` — 442 patients with ten baseline measurements (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`) and `progression`, a quantitative measure of disease progression after one year. No missing values and no duplicate rows. The `sex` column is stored as two codes (1 and 2) in the source data, so the script calls them "Group 1" and "Group 2".

## What the script does
1. `info()`, `describe()` and a duplicate check.
2. Builds `bmi_class` (WHO cut-offs) and `progression_level` (above/below the median of 140.5).
3. Compares the share of high-progression patients among obese and non-obese patients.
4. Cross-tabulates BMI class against progression level with totals.
5. Mean and standard deviation of age for high versus low progression.
6. Histogram of age, then horizontal and vertical bar charts of the real group counts.
7. Median progression of obese patients versus everyone.

## Results (from `output.txt`)
* 442 rows, 12 columns, average progression 152.1.
* BMI classes: underweight 2, normal 186, overweight 155, obese 99.
* **79.80 %** of obese patients have high progression, against **41.40 %** of the other patients.
* Age: high progression 50.58 ± 12.99 years, low progression 46.45 ± 12.93 years.
* Median progression: 230 for obese patients, 140.5 overall.
* Group sizes: Group 1 = 235, Group 2 = 207.

This is exploratory work, so it shows an association between BMI and progression, not a cause.

## How to run
```bash
pip install pandas matplotlib
python practical_03.py
```
Charts are saved in `figures/`.

## Files
`practical_03.py` · `practical_03.ipynb` · `output.txt` · `figures/` · `README.md`
