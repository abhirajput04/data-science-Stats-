# Practical 8 — Decision Tree Classification

## Aim
Train and evaluate a decision-tree classifier, compare the entropy and Gini criteria at several depths, read the tree, and verify the first split by hand.

## Data
`Datasets/StudentsPerformance.csv` — 1000 students. **Target:** free/reduced lunch (1) vs standard lunch (0) — 355 vs 645 students, so always guessing "standard" is already 64.5 % accurate. **Features:** three exam scores, gender, ethnic group, parental education and test-preparation course (14 columns after one-hot encoding). This is an educational exercise, not a tool for judging real students.

## What the script does
1. Stratified 80/20 split (800 train, 200 test, seed 11).
2. Mutual information and correlation of each feature with the target.
3. Entropy and Gini trees at depths 2–6 and unlimited, scored on train data, 5-fold cross-validation and test data.
4. Picks the setting with the best cross-validated accuracy on the training data (not the test set).
5. Confusion matrix and classification report, tree plot, `export_text` rules, feature-importance chart.
6. Entropy, Gini and information gain computed manually for every feature, compared with scikit-learn's root split.

## Results (from `output.txt`)
* Strongest features: `math_score` (mutual information 0.069, correlation −0.37), then `reading_score` and `writing_score`.
* Cross-validation picks **Gini, max_depth = 3** (CV accuracy 0.699).
* Train accuracy 0.718, **test accuracy 0.670**.
* Confusion matrix `[[119, 10], [56, 15]]`: standard-lunch recall 0.92, free/reduced recall only 0.21.
* An unlimited tree reaches 99.9 % training accuracy but only about 59–60 % test accuracy — a clear case of overfitting.
* Root split: `math_score <= 50.5`, information gain 0.0681, Gini reduction 0.0451 — identical in the hand calculation and in scikit-learn.

The tree is only a little better than the 64.5 % baseline and finds few free/reduced cases, so exam scores alone say little about lunch type.

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn
python practical_08.py
```

## Files
`practical_08.py` · `practical_08.ipynb` · `output.txt` · `figures/` · `README.md`
