# %% [markdown]
# # Practical 4 - Data Visualisation with Matplotlib
# Dataset: Breast Cancer Wisconsin (569 tumour samples, measurements from cell-nucleus images)

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

DATA = find_data("breast_cancer.csv")
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

tumours = pd.read_csv(DATA / "breast_cancer.csv")
focus = ["mean radius", "mean texture", "mean area", "mean concavity", "diagnosis"]
print("Full table shape :", tumours.shape)
print(tumours[focus].head().to_string())

# %% [markdown]
# ## 1. Checking for duplicate records

# %%
exact = tumours.duplicated().sum()
by_profile = tumours.duplicated(subset=["mean radius", "mean texture", "mean area"]).sum()
cleaned = tumours.drop_duplicates(subset=["mean radius", "mean texture", "mean area"])
print("Exact duplicate rows            :", int(exact))
print("Duplicates on radius/texture/area:", int(by_profile))
print("Rows before / after cleaning    :", len(tumours), "/", len(cleaned))

# %% [markdown]
# ## 2. Summary statistics of the selected measurements

# %%
print(cleaned[focus[:-1]].describe().round(3).to_string())
print("\nSamples per diagnosis:")
print(cleaned["diagnosis"].value_counts().to_string())

# %% [markdown]
# ## 3. Average tumour area by diagnosis (bar chart)

# %%
area_by_dx = cleaned.groupby("diagnosis")["mean area"].agg(["mean", "std"])
print(area_by_dx.round(2).to_string())

plt.figure(figsize=(6, 4.5))
plt.bar(area_by_dx.index, area_by_dx["mean"], yerr=area_by_dx["std"], capsize=8,
        color=["#c0392b", "#27ae60"])
plt.title("Average Tumour Area by Diagnosis")
plt.ylabel("Mean area (pixels)")
plt.savefig(FIG / "bar_area_by_diagnosis.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Radius against concavity (all samples)

# %%
plt.figure(figsize=(7, 5))
plt.scatter(cleaned["mean radius"], cleaned["mean concavity"], s=18, alpha=0.6, color="#34495e")
plt.title("Mean Radius vs Mean Concavity")
plt.xlabel("Mean radius")
plt.ylabel("Mean concavity")
plt.savefig(FIG / "scatter_radius_concavity.png", dpi=120, bbox_inches="tight")
plt.show()
r = cleaned["mean radius"].corr(cleaned["mean concavity"])
print(f"Correlation between radius and concavity: {r:.3f}")

# %% [markdown]
# ## 5. Same scatter, coloured by diagnosis

# %%
palette = {"malignant": "#c0392b", "benign": "#27ae60"}
plt.figure(figsize=(7, 5))
for label, part in cleaned.groupby("diagnosis"):
    plt.scatter(part["mean radius"], part["mean concavity"], s=18, alpha=0.65,
                color=palette[label], label=label)
plt.title("Radius vs Concavity by Diagnosis")
plt.xlabel("Mean radius")
plt.ylabel("Mean concavity")
plt.legend()
plt.savefig(FIG / "scatter_by_diagnosis.png", dpi=120, bbox_inches="tight")
plt.show()

for label, part in cleaned.groupby("diagnosis"):
    print(f"{label:<10} mean radius = {part['mean radius'].mean():.2f}, mean concavity = {part['mean concavity'].mean():.3f}")

# %% [markdown]
# ## 6. Four groups: diagnosis x texture level

# %%
texture_cut = cleaned["mean texture"].median()
grouped = cleaned.assign(texture_level=np.where(cleaned["mean texture"] >= texture_cut, "rough", "smooth"))

markers = {"smooth": "o", "rough": "^"}
plt.figure(figsize=(7.5, 5.5))
for (dx, tex), part in grouped.groupby(["diagnosis", "texture_level"]):
    plt.scatter(part["mean radius"], part["mean concavity"], s=26, alpha=0.7,
                color=palette[dx], marker=markers[tex], label=f"{dx} / {tex}")
plt.title("Radius vs Concavity by Diagnosis and Texture Level")
plt.xlabel("Mean radius")
plt.ylabel("Mean concavity")
plt.legend()
plt.savefig(FIG / "scatter_four_groups.png", dpi=120, bbox_inches="tight")
plt.show()

print(grouped.groupby(["diagnosis", "texture_level"]).size().rename("samples").to_string())
