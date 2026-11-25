# %% [markdown]
# # Practical 3 - Exploratory Data Analysis
# Dataset: Diabetes progression - 442 patients, baseline measurements and disease progression after one year

# %%
from pathlib import Path
import io
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

DATA = find_data("diabetes_progression.csv")
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

patients = pd.read_csv(DATA / "diabetes_progression.csv")
print(patients.head().to_string())

# %% [markdown]
# ## 1. Structure of the data

# %%
buffer = io.StringIO()
patients.info(buf=buffer)
print(buffer.getvalue())
print("Duplicate rows:", int(patients.duplicated().sum()))

# %%
print(patients.drop(columns="patient_id").describe().round(2).to_string())

# %% [markdown]
# ## 2. Creating categories
# BMI classes follow the usual WHO cut-offs; "high progression" means above the overall median.

# %%
patients["bmi_class"] = pd.cut(patients["bmi"], bins=[0, 18.5, 25, 30, 100],
                               labels=["underweight", "normal", "overweight", "obese"], right=False)
median_prog = patients["progression"].median()
patients["progression_level"] = (patients["progression"] > median_prog).map({True: "high", False: "low"})
print("Median progression:", median_prog)
print(patients["bmi_class"].value_counts().reindex(["underweight", "normal", "overweight", "obese"]).to_string())

# %% [markdown]
# ## 3. How many obese patients show high progression?

# %%
obese = patients[patients["bmi_class"] == "obese"]
share_obese = (obese["progression_level"] == "high").mean() * 100
rest = patients[patients["bmi_class"] != "obese"]
share_rest = (rest["progression_level"] == "high").mean() * 100

print(f"Obese patients               : {len(obese)}")
print(f"  with high progression      : {share_obese:.2f}%")
print(f"Non-obese patients           : {len(rest)}")
print(f"  with high progression      : {share_rest:.2f}%")
print("Conclusion: progression differs clearly with BMI" if abs(share_obese - share_rest) > 10
      else "Conclusion: little difference")

# %% [markdown]
# ## 4. BMI class vs progression level

# %%
print(pd.crosstab(patients["bmi_class"], patients["progression_level"], margins=True).to_string())

# %% [markdown]
# ## 5. Age of high- versus low-progression patients

# %%
age_stats = patients.groupby("progression_level")["age"].agg(["count", "mean", "std"])
print(age_stats.round(2).to_string())

# %% [markdown]
# ## 6. Age distribution

# %%
plt.figure(figsize=(7, 4.5))
plt.hist(patients["age"], bins=12, color="#16a085", edgecolor="white")
plt.title("Age of Patients")
plt.xlabel("Age (years)")
plt.ylabel("Number of patients")
plt.savefig(FIG / "age_histogram.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Bar charts of real group counts

# %%
sex_counts = patients["sex"].map({1.0: "Group 1", 2.0: "Group 2"}).value_counts().sort_index()
print(sex_counts.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(sex_counts.index, sex_counts.values, color=["#2980b9", "#e67e22"])
axes[0].set_title("Horizontal bar chart")
axes[0].set_xlabel("Patients")
axes[1].bar(sex_counts.index, sex_counts.values, color=["#2980b9", "#e67e22"])
axes[1].set_title("Vertical bar chart")
axes[1].set_ylabel("Patients")
fig.savefig(FIG / "group_counts.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Median progression of obese patients

# %%
print("Median progression (obese)    :", obese["progression"].median())
print("Median progression (all)      :", median_prog)
