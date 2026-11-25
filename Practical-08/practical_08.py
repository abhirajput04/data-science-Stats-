# %% [markdown]
# # Practical 8 - Decision Tree Classification
# Dataset: Students Performance (1000 students) - predict whether a student receives a
# free/reduced lunch from exam scores and background information.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif

SEED = 11
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

DATA = find_data("StudentsPerformance.csv")
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "StudentsPerformance.csv")
raw.columns = [c.replace(" ", "_").replace("/", "_") for c in raw.columns]

y = (raw["lunch"] == "free/reduced").astype(int)          # 1 = free/reduced lunch
X = pd.get_dummies(raw.drop(columns="lunch"), drop_first=True, dtype=int)

print("Students           :", len(raw))
print("Standard lunch     :", int((y == 0).sum()))
print("Free/reduced lunch :", int((y == 1).sum()))
print("Majority-class baseline accuracy: %.3f" % (1 - y.mean()))
print("Number of features after encoding:", X.shape[1])

# %% [markdown]
# ## 1. Train / test split (80 / 20, stratified)

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print("Training set:", X_tr.shape, "| Test set:", X_te.shape)
print("Share of free/reduced lunch  train: %.3f  test: %.3f" % (y_tr.mean(), y_te.mean()))

# %% [markdown]
# ## 2. Which features look informative?

# %%
mi = pd.Series(mutual_info_classif(X_tr, y_tr, random_state=SEED), index=X.columns)
corr = X_tr.assign(target=y_tr).corr()["target"].drop("target")
ranking = pd.DataFrame({"mutual_info": mi, "correlation": corr}).sort_values("mutual_info", ascending=False)
print(ranking.round(4).head(8).to_string())

# %% [markdown]
# ## 3. Entropy vs Gini at different depths

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
records = []
for criterion in ("entropy", "gini"):
    for depth in (2, 3, 4, 5, 6, None):
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=SEED)
        cv_acc = cross_val_score(tree, X_tr, y_tr, cv=cv).mean()
        tree.fit(X_tr, y_tr)
        records.append({"criterion": criterion, "max_depth": depth if depth else 0,
                        "train_acc": tree.score(X_tr, y_tr), "cv_acc": cv_acc,
                        "test_acc": tree.score(X_te, y_te)})
sweep = pd.DataFrame(records)
print("(max_depth 0 means 'no limit')")
print(sweep.round(3).to_string(index=False))

limited = sweep[sweep["max_depth"] > 0]
best = limited.sort_values(["cv_acc", "max_depth"], ascending=[False, True]).iloc[0]
print(f"\nChosen by 5-fold cross-validation on the training data: {best['criterion']}, "
      f"max_depth = {int(best['max_depth'])}")

# %% [markdown]
# ## 4. Evaluating the chosen tree

# %%
model = DecisionTreeClassifier(criterion=best["criterion"], max_depth=int(best["max_depth"]),
                               random_state=SEED).fit(X_tr, y_tr)
pred = model.predict(X_te)

print(f"Training accuracy: {model.score(X_tr, y_tr):.3f}")
print(f"Test accuracy    : {accuracy_score(y_te, pred):.3f}")
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_te, pred))
print()
print(classification_report(y_te, pred, target_names=["standard", "free/reduced"]))

# %% [markdown]
# ## 5. Visualising the tree

# %%
plt.figure(figsize=(20, 9))
plot_tree(model, feature_names=list(X.columns), class_names=["standard", "free/reduced"],
          filled=True, rounded=True, fontsize=9)
plt.title(f"Decision tree ({best['criterion']}, max_depth={int(best['max_depth'])})")
plt.savefig(FIG / "decision_tree.png", dpi=110, bbox_inches="tight")
plt.show()
print(export_text(model, feature_names=list(X.columns)))

# %%
importance = pd.Series(model.feature_importances_, index=X.columns)
importance[importance > 0].sort_values().plot(kind="barh", figsize=(7, 4), color="#2c7fb8")
plt.title("Feature importance")
plt.xlabel("Importance")
plt.savefig(FIG / "feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Entropy, information gain and Gini calculated by hand

# %%
def entropy(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) + 0.0

def gini(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    return float(1 - (p ** 2).sum())

def best_threshold(values, labels):
    """Return (information gain, threshold) of the best binary split on one feature."""
    order = np.argsort(values)
    v, l = values[order], labels[order]
    base, top = entropy(l), (0.0, None)
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            continue
        w = i / len(v)
        gain = base - (w * entropy(l[:i]) + (1 - w) * entropy(l[i:]))
        if gain > top[0]:
            top = (gain, (v[i] + v[i - 1]) / 2)
    return top

labels = y_tr.to_numpy()
print(f"Entropy of the training labels: {entropy(labels):.4f}")
print(f"Gini of the training labels   : {gini(labels):.4f}\n")

scan = pd.DataFrame(
    [(col, *best_threshold(X_tr[col].to_numpy(dtype=float), labels)) for col in X.columns],
    columns=["feature", "info_gain", "threshold"]).sort_values("info_gain", ascending=False)
print(scan.round(4).head(6).to_string(index=False))

# %%
stump = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=SEED).fit(X_tr, y_tr)
root_feature = X.columns[stump.tree_.feature[0]]
root_threshold = stump.tree_.threshold[0]
print(f"scikit-learn root split : {root_feature} <= {root_threshold:.2f}")
print(f"My calculation          : {scan.iloc[0]['feature']} <= {scan.iloc[0]['threshold']:.2f}")

mask = X_tr[root_feature].to_numpy() <= root_threshold
left, right = labels[mask], labels[~mask]
w = len(left) / len(labels)
child_entropy = w * entropy(left) + (1 - w) * entropy(right)
child_gini = w * gini(left) + (1 - w) * gini(right)
print(f"\nLeft child : n={len(left)}  (free/reduced={int(left.sum())}), entropy={entropy(left):.4f}, gini={gini(left):.4f}")
print(f"Right child: n={len(right)}  (free/reduced={int(right.sum())}), entropy={entropy(right):.4f}, gini={gini(right):.4f}")
print(f"Information gain = {entropy(labels):.4f} - {child_entropy:.4f} = {entropy(labels) - child_entropy:.4f}")
print(f"Gini reduction   = {gini(labels):.4f} - {child_gini:.4f} = {gini(labels) - child_gini:.4f}")
