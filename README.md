# Data Science Practicals 1–10 (rewritten)

Every practical keeps the same learning goal and the same style of output as before, but the code, the data and the numbers are new.

| # | Topic | Dataset | Main tools |
|---|---|---|---|
| 1 | Matrix operations, one-hot encoding | Teaching Ratings | Pandas, NumPy |
| 2 | Descriptive statistics | Wine | Pandas, Matplotlib |
| 3 | Exploratory data analysis | Diabetes progression | Pandas, Matplotlib |
| 4 | Data visualisation | Breast Cancer Wisconsin | Pandas, Matplotlib |
| 5 | Probability and Z-test | Iris + delivery-time scenario | SciPy |
| 6 | t-test, ANOVA, Chi-square, correlation | Students Performance | SciPy |
| 7 | Regression and ANOVA | Diabetes progression | NumPy, SciPy |
| 8 | Decision tree | Students Performance | scikit-learn |
| 9 | K-Nearest Neighbours | Wine | scikit-learn |
| 10 | PCA | Breast Cancer Wisconsin | scikit-learn |

## Folder layout
```
Datasets/            all CSV files + README
Practical-NN/
    practical_NN.py      runnable script (cells marked with # %%)
    practical_NN.ipynb   same code, already executed, outputs included
    output.txt           console output of the script
    figures/             saved charts
    README.md            aim, method, results
requirements.txt
```

## Running
```bash
pip install -r requirements.txt
cd Practical-06
python practical_06.py
```
Scripts find the data in `../Datasets`, `./Datasets` or the current folder, so they also run in Google Colab after uploading the CSV. Every `output.txt` and notebook output comes from a real run of the code in this folder.

## Google Colab
1. Open `practical_NN.ipynb` in Colab.
2. In the left sidebar choose **Files**, then upload the CSV the practical needs (see `Datasets/README.md`). It must land in `/content`.
3. Run all cells. If the CSV is missing, the first cell opens an upload dialog and asks for it.
Error `FileNotFoundError: Could not find ...csv` simply means the CSV was not uploaded yet.
