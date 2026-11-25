# Practical 1 — Matrix Operations with Pandas and NumPy

## Aim
Load a real table with Pandas, turn part of it into NumPy arrays, and practise the basic array and matrix operations, finishing with one-hot encoding of the text columns.

## Data
`Datasets/TeachingRatings.csv` — 463 course evaluations of 94 instructors (13 columns, no missing values).
Numeric columns used for the matrices: `age`, `beauty`, `eval`, `students`.

## What the script does
1. Reads the CSV and prints the shape, column types and missing-value count.
2. Converts four numeric columns to a `float64` array and reports shape, size, `ndim`, dtype and memory use.
3. Takes rows 4–7 of `age` and `students` and splits them into two 2×2 matrices **P** and **Q**.
4. Performs `+`, `-`, element-wise `*` and `/`, the matrix product (`@`) in both orders, the transpose, determinant and rank.
5. Inverts **P** with a small guard against singular matrices and proves `P @ P⁻¹ = I`.
6. Computes column mean, standard deviation, minimum and maximum directly from the array.
7. One-hot encodes the six text columns with `pd.get_dummies`.

## Results (from `output.txt`)
| Item | Value |
|---|---|
| Table shape | 463 × 13 |
| Numeric array | shape (463, 4), size 1852, 2 dimensions |
| P (rows 4–5) | [[31, 42], [62, 182]] |
| Q (rows 6–7) | [[33, 33], [51, 25]] |
| P + Q | [[64, 75], [113, 207]] |
| P @ Q | [[3165, 2073], [11328, 6596]] |
| det(P), rank(P) | 3038, 2 |
| P⁻¹ | [[0.0599, −0.0138], [−0.0204, 0.0102]] |
| Shape after one-hot encoding | 463 × 19 |

`P @ Q` and `Q @ P` are different, so matrix multiplication is not commutative. The determinant is non-zero, which is why P can be inverted.

## How to run
```bash
pip install pandas numpy
python practical_01.py          # or open practical_01.ipynb
```
The script looks for the CSV in `../Datasets`, then `./Datasets`, then the current folder, so it also works after uploading the file to Google Colab.

## Files
`practical_01.py` · `practical_01.ipynb` (executed, with outputs) · `output.txt` · `README.md`
