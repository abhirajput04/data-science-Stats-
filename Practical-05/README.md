# Practical 5 — Probability and Hypothesis Testing

## Aim
Compare probabilities predicted by a normal model with probabilities counted in real data, then run a two-tailed Z-test.

## Data
* `Datasets/iris.csv` — sepal length of 150 iris flowers (mean 5.8433 cm, sample sd 0.8281 cm).
* Z-test scenario (given values, not from a file): a courier claims an average delivery time of 32 minutes, known σ = 6 min; 45 recent deliveries averaged 30.4 minutes; α = 0.05.

## What the script does
* A helper `compare(lower, upper)` returns the theoretical probability (`scipy.stats.norm`) and the observed proportion.
* **Q1:** P(sepal length > 6.5 cm).
* **Q2:** P(5.0 < sepal length < 6.0 cm).
* **Q3:** `two_tailed_z_test()` returns standard error, z, p-value, critical value, 95 % confidence interval and the decision; a plot shows the rejection regions.

## Results (from `output.txt`)
| Question | z | Theoretical | Observed |
|---|---|---|---|
| Q1: X > 6.5 | 0.7930 | 21.39 % | 20.00 % |
| Q2: 5.0 < X < 6.0 | −1.0184 and 0.1892 | 42.08 % | 34.00 % |

Z-test: SE = 0.8944, z = −1.7889, p = 0.0736, critical value ±1.96, 95 % CI (28.65, 32.15).
**Decision: fail to reject H0** — p is above 0.05, so there is no significant evidence that the average delivery time has changed.

The normal model is close for the upper tail but overestimates the middle band by about 8 percentage points, because real sepal lengths are not perfectly bell-shaped.

## How to run
```bash
pip install pandas numpy scipy matplotlib
python practical_05.py
```

## Files
`practical_05.py` · `practical_05.ipynb` · `output.txt` · `figures/z_test.png` · `README.md`
