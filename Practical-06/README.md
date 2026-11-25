# Practical 6 — Statistical Tests (t-test, ANOVA, Chi-square, Correlation)

## Aim
Choose and run the right test for four kinds of question, and read the p-value together with an effect size.

## Data
`Datasets/StudentsPerformance.csv` — 1000 students: gender, ethnic group (A–E), parental education, lunch type, test-preparation course, and math / reading / writing scores. Significance level α = 0.05 throughout.

## Tests performed
| # | Question | Test |
|---|---|---|
| Q1 | Does the test-preparation course change the writing score? | Welch independent-samples t-test + Cohen's d |
| Q2 | Does the reading score differ between the five groups? | One-way ANOVA + eta squared + Tukey HSD |
| Q3 | Is gender associated with completing the course? | Chi-square test of independence + Cramér's V |
| Q4 | Are reading and writing scores correlated? | Pearson r (with Spearman ρ as a check) |

## Results (from `output.txt`)
* **Q1** — completed 74.42 (n = 358) vs none 64.50 (n = 642); t = 10.75, p = 2.7e-25, Cohen's d = 0.70 → **reject H0**, a medium-to-large effect.
* **Q2** — group means rise from 64.67 (A) to 73.03 (E); F = 5.62, p = 0.00018, η² = 0.022 → **reject H0**. Tukey shows A–D, A–E and B–E differ. The effect is small.
* **Q3** — χ² = 0.0155, df = 1, p = 0.9008, Cramér's V = 0.004 → **fail to reject H0**; gender and course completion are unrelated here.
* **Q4** — r = 0.9546, ρ = 0.9490, p below 1e-300 → **reject H0**; a very strong positive relationship.

A tiny p-value alone does not mean a big effect — compare Q2 (significant but η² of only 2 %) with Q1 and Q4.

## How to run
```bash
pip install pandas numpy scipy matplotlib
python practical_06.py
```
Box plots, a stacked bar chart and a scatter plot with fitted line are saved in `figures/`.

## Files
`practical_06.py` · `practical_06.ipynb` · `output.txt` · `figures/` · `README.md`
