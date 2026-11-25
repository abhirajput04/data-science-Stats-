# Practical 4 — Data Visualisation with Matplotlib

## Aim
Check data quality, summarise it, and tell the story with four kinds of chart: bar chart with error bars, plain scatter plot, scatter by one group variable and scatter by two group variables.

## Data
`Datasets/breast_cancer.csv` — 569 breast-mass samples (357 benign, 212 malignant) described by 30 measurements taken from cell-nucleus images. The script focuses on `mean radius`, `mean texture`, `mean area` and `mean concavity`.

## What the script does
1. Looks for exact duplicate rows and for duplicates on (radius, texture, area), then applies `drop_duplicates`.
2. Summary statistics of the four measurements.
3. **Bar chart** of average tumour area per diagnosis with standard-deviation error bars.
4. **Scatter plot** of mean radius against mean concavity.
5. The same scatter **coloured by diagnosis**.
6. A four-group scatter: diagnosis × texture level (above/below median texture) with different colours and marker shapes.

## Results (from `output.txt`)
* No duplicates were found, so 569 rows remain after cleaning.
* Mean area: benign 462.8 (sd 134.3), malignant 978.4 (sd 367.9).
* Radius and concavity are positively correlated (r = 0.677).
* Malignant samples have larger radius (17.46 vs 12.15) and higher concavity (0.161 vs 0.046).
* Group sizes: benign/rough 119, benign/smooth 238, malignant/rough 166, malignant/smooth 46 — malignant samples are much more often "rough".

## How to run
```bash
pip install pandas numpy matplotlib
python practical_04.py
```
The four charts are saved in `figures/`.

## Files
`practical_04.py` · `practical_04.ipynb` · `output.txt` · `figures/` · `README.md`
