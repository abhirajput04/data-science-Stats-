# Practical 7 — Regression Analysis and ANOVA

## Aim
Fit simple regression models, run a one-way ANOVA, and interpret coefficients, R² and p-values.

## Data
`Datasets/diabetes_progression.csv` — 442 patients; outcome `progression` (mean 152.1). Predictors used: blood pressure (`bp`), `bmi` and `age`.

## Method note
The regression and ANOVA tables are built **from first principles** with NumPy and SciPy (`ols()` and `anova_oneway()` in the script): coefficients from least squares, standard errors from σ²(XᵀX)⁻¹, t and F p-values from SciPy distributions. Each result is cross-checked against SciPy (`stats.linregress`, `stats.f_oneway`) and the numbers match. If you prefer `statsmodels`, the same models are `smf.ols("progression ~ C(bp_level)", df)`, `smf.ols("progression ~ bmi", df)` and `anova_lm(smf.ols("bmi ~ C(age_group)", df))`.

## Questions and results (from `output.txt`)
**Q1 — progression ~ blood-pressure level** (high = above median bp)
* Intercept 126.88; high-BP coefficient **+53.40** (t = 7.74, p = 6.9e-14).
* R² = 0.1199, F = 59.92 → **reject H0**: high-BP patients progress about 53 points more.

**Q2 — ANOVA: does BMI differ across age groups?** (under 40 / 40–55 / over 55)
| Group | n | mean BMI |
|---|---|---|
| under 40 | 122 | 25.385 |
| 40–55 | 177 | 26.346 |
| over 55 | 143 | 27.258 |

F = 6.057, p = 0.0025 → **reject H0**; mean BMI rises with age.

**Q3 — progression ~ BMI**
* Intercept −117.77; BMI slope **10.23** (t = 15.19, p = 3.5e-42).
* R² = 0.3439, F = 230.65 → **reject H0**; each extra BMI point goes with about 10 more progression points.

BMI explains 34 % of the variation, blood-pressure level only 12 %, so most variation comes from other factors.

## How to run
```bash
pip install pandas numpy scipy matplotlib
python practical_07.py
```

## Files
`practical_07.py` · `practical_07.ipynb` · `output.txt` · `figures/` · `README.md`
