# %% [markdown]
# # Practical 1 - Matrix Operations with Pandas and NumPy
# Dataset: Teaching Ratings (463 course evaluations of 94 instructors)

# %%
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
def find_data(name):
    """Look for the CSV in the usual places; in Colab, ask for an upload if it is missing."""
    folders = (HERE.parent / "Datasets", HERE / "Datasets", HERE, Path.cwd(),
               Path("/content"), Path("/content/Datasets"), Path("/content/drive/MyDrive"))
    for folder in folders:
        if (folder / name).exists():
            return folder
    try:
        from google.colab import files
        print(f"{name} not found - please choose it in the upload dialog (it is in the Datasets folder).")
        files.upload()
        return Path.cwd()
    except ImportError:
        raise FileNotFoundError(f"Could not find {name}. Put it in a 'Datasets' folder next to this script.")

DATA = find_data("TeachingRatings.csv")

ratings = pd.read_csv(DATA / "TeachingRatings.csv")
np.set_printoptions(precision=4, suppress=True)

print("Rows x Columns :", ratings.shape)
print("\nColumn types:")
print(ratings.dtypes.to_string())
print("\nMissing values per column:", int(ratings.isna().sum().sum()))

# %% [markdown]
# ## 1. From DataFrame to NumPy array

# %%
numeric_cols = ["age", "beauty", "eval", "students"]
data = ratings[numeric_cols].to_numpy(dtype=float)

print("Array shape      :", data.shape)
print("Number of items  :", data.size)
print("Dimensions (ndim):", data.ndim)
print("Data type        :", data.dtype)
print("Memory used      :", data.nbytes, "bytes")

# %% [markdown]
# ## 2. Pick two small blocks of rows and form square matrices

# %%
subset = ratings.iloc[4:8][["age", "students"]]
print("Selected rows (4 to 7), columns age & students:")
print(subset.to_string())

block = subset.to_numpy(dtype=float)
P, Q = block[:2], block[2:]
print("\nMatrix P (rows 4-5):\n", P)
print("\nMatrix Q (rows 6-7):\n", Q)

# %% [markdown]
# ## 3. Element-wise operations

# %%
operations = {
    "P + Q": P + Q,
    "P - Q": P - Q,
    "P * Q (element-wise)": P * Q,
    "P / Q (element-wise)": P / Q,
}
for label, result in operations.items():
    print(f"{label}:\n{result}\n")

# %% [markdown]
# ## 4. Linear-algebra operations

# %%
print("Matrix product P @ Q:\n", P @ Q)
print("\nMatrix product Q @ P:\n", Q @ P)
print("\nTranspose of P:\n", P.T)
print("\nDeterminant of P :", round(np.linalg.det(P), 4))
print("Rank of P        :", np.linalg.matrix_rank(P))

# %% [markdown]
# ## 5. Inverse and verification

# %%
def safe_inverse(M):
    if abs(np.linalg.det(M)) < 1e-12:
        raise ValueError("Matrix is singular, no inverse exists")
    return np.linalg.inv(M)

P_inv = safe_inverse(P)
print("Inverse of P:\n", P_inv)
print("\nCheck P @ P_inv (should be identity):\n", np.round(P @ P_inv, 6))
print("Identity recovered:", np.allclose(P @ P_inv, np.eye(2)))

# %% [markdown]
# ## 6. Column statistics straight from the array

# %%
summary = pd.DataFrame({
    "mean": data.mean(axis=0),
    "std": data.std(axis=0, ddof=1),
    "min": data.min(axis=0),
    "max": data.max(axis=0),
}, index=numeric_cols)
print(summary.round(3).to_string())

# %% [markdown]
# ## 7. Encoding the text columns (one-hot)

# %%
text_cols = ratings.select_dtypes(exclude="number").columns.tolist()
print("Categorical columns:", text_cols)

encoded = pd.get_dummies(ratings, columns=text_cols, dtype=int)
print("\nShape before encoding:", ratings.shape)
print("Shape after encoding :", encoded.shape)
print("\nFirst 3 rows of the encoded columns:")
dummy_cols = [c for c in encoded.columns if c not in ratings.columns]
print(encoded[dummy_cols].head(3).to_string())
