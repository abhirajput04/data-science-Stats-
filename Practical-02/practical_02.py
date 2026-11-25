# %% [markdown]
# # Practical 2 - Foundations of Statistics (Descriptive Statistics)
# Dataset: Wine chemistry - 178 Italian wines from three cultivars

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

DATA = find_data("wine.csv")
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

wine = pd.read_csv(DATA / "wine.csv")
print("Shape            :", wine.shape)
print("Columns          :", list(wine.columns))
print("Missing values   :", int(wine.isna().sum().sum()))
print("Wines per class  :")
print(wine["wine_class"].value_counts().to_string())

# %% [markdown]
# ## 1. What kind of dataset is this?
# Every row is a different wine measured once, there is no time column, so the data is **cross-sectional**.

# %%
has_time_axis = any("date" in c.lower() or "year" in c.lower() for c in wine.columns)
print("Time column present :", has_time_axis)
print("Dataset type        :", "time series / panel" if has_time_axis else "cross-sectional")

# %% [markdown]
# ## 2. Centre and spread of one variable (proline)

# %%
proline = wine["proline"]
centre_spread = {
    "mean": proline.mean(),
    "median": proline.median(),
    "minimum": proline.min(),
    "maximum": proline.max(),
    "range": proline.max() - proline.min(),
    "std dev": proline.std(),
    "IQR": proline.quantile(0.75) - proline.quantile(0.25),
}
for name, value in centre_spread.items():
    print(f"{name:<8}: {value:10.3f}")

# %% [markdown]
# ## 3. Descriptive-statistics table

# %%
table = wine.drop(columns="wine_class").describe().T
table["cv_%"] = 100 * table["std"] / table["mean"]
print(table[["count", "mean", "std", "min", "50%", "max", "cv_%"]].round(3).to_string())

# %% [markdown]
# ## 4. Shape of the alcohol distribution

# %%
alc = wine["alcohol"]
print(f"Mean     : {alc.mean():.4f}")
print(f"Median   : {alc.median():.4f}")
print(f"Std dev  : {alc.std():.4f}")
print(f"Skewness : {alc.skew():.4f}")
print(f"Kurtosis : {alc.kurt():.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(alc, bins=14, color="#8e44ad", edgecolor="white")
ax.axvline(alc.mean(), color="black", linestyle="--", label=f"mean {alc.mean():.2f}")
ax.axvline(alc.median(), color="orange", linestyle=":", linewidth=2, label=f"median {alc.median():.2f}")
ax.set(title="Distribution of Alcohol Content", xlabel="Alcohol (% vol)", ylabel="Number of wines")
ax.legend()
fig.savefig(FIG / "alcohol_histogram.png", dpi=120, bbox_inches="tight")
plt.show()

skew = alc.skew()
shape = "roughly symmetric" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
print("Comment: the alcohol distribution is", shape)

# %% [markdown]
# ## 5. Group-wise statistics

# %%
by_class = wine.groupby("wine_class").agg(
    wines=("alcohol", "size"),
    alcohol_mean=("alcohol", "mean"),
    alcohol_std=("alcohol", "std"),
    flavanoids_mean=("flavanoids", "mean"),
    colour_mean=("color_intensity", "mean"),
)
print(by_class.round(3).to_string())

# %% [markdown]
# ## 6. Cross-tabulation: wine class vs alcohol level

# %%
cutoff = wine["alcohol"].median()
wine["alcohol_level"] = np.where(wine["alcohol"] >= cutoff, "high", "low")
print(f"(median alcohol used as cut-off: {cutoff})\n")

counts = pd.crosstab(wine["wine_class"], wine["alcohol_level"])
print("Counts:")
print(counts.to_string())

percent = pd.crosstab(wine["wine_class"], wine["alcohol_level"], normalize="index") * 100
print("\nRow percentages:")
print(percent.round(2).to_string())

# %% [markdown]
# ## 7. Spotting outliers with the 1.5 x IQR rule

# %%
q1, q3 = wine["magnesium"].quantile([0.25, 0.75])
low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
flagged = wine[(wine["magnesium"] < low) | (wine["magnesium"] > high)]
print(f"Magnesium fence: [{low:.1f}, {high:.1f}]")
print("Outlying wines  :", len(flagged))
print(flagged[["wine_class", "magnesium"]].to_string())
