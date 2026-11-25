# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.2 · A column transformer for the screen
#
# Day 3 · Session 2 (Encoding, scaling and missing values) · 35 min.
#
# Models read numbers. The sector is a word, body mass index (BMI) is missing for some clients, and the walk is
# in seconds while ages are in years. You encode, impute and scale, each step learnt from the training rows only,
# and you watch scaling change the answer of a neighbour model.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/assessments.csv` and the fixed split. Nothing from Lab 3.1.
#
# **Tasks**
# - [ ] 1 · Find the categorical, numeric and incomplete columns (5')
# - [ ] 2 · One-hot encode the sector (6')
# - [ ] 3 · Impute body mass index from training rows and add a missing flag (8')
# - [ ] 4 · Scale the numbers and compare a neighbour model with and without (10')
# - [ ] 5 · Put it all in one column transformer inside a pipeline (6')
# - [ ] ✱ Stretch · Compare the missing rate of body mass index for clients who live alone and with others

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/fnd-aic-101.git"
    if not Path("/content/fnd-aic-101").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/fnd-aic-101"], check=True)
    os.chdir("/content/fnd-aic-101/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("hurbuth.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import hurbuth as U
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

train, val, _ = U.split()
print(train.drop(columns=["assessment_id", "client_id"]).head(3).to_string())

# %% [markdown]
# ## 1 · Three kinds of column (5')
#
# ✏️ **Exercise** · Write `column_kinds(df, candidates)`: from the list `candidates` return a dictionary with
# `"categorical"` (columns whose values are text), `"numeric"` (the others) and `"incomplete"` (columns with at
# least one missing value).

# %%
CANDIDATES = U.FEATURES + ["bmi", "sector"]


def column_kinds(df: pd.DataFrame, candidates: list[str]) -> dict:
    # BEGIN SOLUTION
    cat = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])]
    return {"categorical": cat, "numeric": [c for c in candidates if c not in cat],
            "incomplete": [c for c in candidates if df[c].isna().any()]}
    # END SOLUTION


# %%
try:
    kinds = column_kinds(train, CANDIDATES)
except NotImplementedError:
    kinds = U.reference("d3_2_kinds")
U.keep("d3_2_kinds", kinds)
print(kinds)
U.check(kinds["categorical"] == ["sector"] and kinds["incomplete"] == ["bmi"], "sector is the text column and bmi the incomplete one")
print(f"bmi missing in {train.bmi.isna().mean():.1%} of training rows")

# %% [markdown]
# `sex` and `nurse_preference` are text columns too, and they stay out of the screen's inputs: sex is a
# sensitive attribute that Hurbuth decided not to use, and Lab 4.3 shows what nurse preference carries.
#
# ## 2 · One-hot encoding (6')
#
# A sector is not a number: "West" is not twice "North". One-hot encoding gives each sector its own 0-or-1
# column. The encoder learns the list of sectors from the training rows; a sector it has never seen gets all
# zeros (`handle_unknown="ignore"`).
#
# ✏️ **Exercise** · Write `encode_sector(train, other)`: fit `OneHotEncoder(handle_unknown="ignore")` on the
# training `sector` column and return the encoded `other` rows as a DataFrame with the encoder's column names.

# %%
def encode_sector(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["sector"]])
    return pd.DataFrame(enc.transform(other[["sector"]]), columns=enc.get_feature_names_out(), index=other.index)
    # END SOLUTION


# %%
try:
    onehot = encode_sector(train, val)
except NotImplementedError:
    onehot = U.reference("d3_2_onehot_head")
U.keep("d3_2_onehot_head", onehot.head())
print(onehot.head().to_string())
U.check((onehot.sum(axis=1) == 1).all(), "every validation row has exactly one sector column set")

# %% [markdown]
# ## 3 · A missing value and a flag (8')
#
# Most models cannot take a blank. Fill BMI with the **training** median, and keep a flag column saying it was
# missing: a blank is information too (Hurbuth records BMI less often for clients who live alone).
#
# ✏️ **Exercise** · Write `impute_bmi(train, other)`: fit `SimpleImputer(strategy="median", add_indicator=True)`
# on the training `bmi` column and return the transformed `other` rows as a DataFrame with the columns `bmi` and
# `bmi_missing`.

# %%
def impute_bmi(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    imp = SimpleImputer(strategy="median", add_indicator=True).fit(train[["bmi"]])
    return pd.DataFrame(imp.transform(other[["bmi"]]), columns=["bmi", "bmi_missing"], index=other.index)
    # END SOLUTION


# %%
try:
    filled = impute_bmi(train, val)
except NotImplementedError:
    filled = U.reference("d3_2_filled_summary")
    print("(reference summary only)")
summary = pd.DataFrame({"rows filled": [int(filled.bmi_missing.sum())], "fill value": [float(filled.bmi[filled.bmi_missing == 1].iloc[0])]}) \
    if "bmi_missing" in filled else filled
U.keep("d3_2_filled_summary", summary)
print(summary.to_string(index=False))
U.check(abs(summary["fill value"].iloc[0] - train.bmi.median()) < 1e-9, "the fill value is the training median")

# %% [markdown]
# ## 4 · Scaling and a neighbour model (10')
#
# Day 1 showed a distance being decided by the column with the biggest numbers. Here are two neighbour models on
# the nine features: one on the raw numbers, one after `StandardScaler` (each column minus its training mean,
# divided by its training standard deviation).
#
# ✏️ **Exercise** · Write `knn_auc(scale)`: fit `KNeighborsClassifier(n_neighbors=50)` on the training rows,
# preceded by `StandardScaler()` when `scale` is true, and return its validation AUC.

# %%
def knn_auc(scale: bool) -> float:
    # BEGIN SOLUTION
    knn = KNeighborsClassifier(n_neighbors=50)
    model = make_pipeline(StandardScaler(), knn) if scale else knn
    model.fit(train[U.FEATURES], train.fell_6m)
    return float(roc_auc_score(val.fell_6m, model.predict_proba(val[U.FEATURES])[:, 1]))
    # END SOLUTION


# %%
try:
    raw_auc, scaled_auc = knn_auc(False), knn_auc(True)
except NotImplementedError:
    raw_auc, scaled_auc = U.reference("d3_2_knn")
U.keep("d3_2_knn", [raw_auc, scaled_auc])
print(f"50 neighbours, validation AUC: raw {raw_auc:.3f}, scaled {scaled_auc:.3f}")
print(train[U.FEATURES].std().round(2).to_string())
U.check(scaled_auc > raw_auc, "scaling helps the neighbour model")

# %% [markdown]
# 🔍 **Look at** the standard deviations: age varies by years, the 0-or-1 columns by about half a unit. Unscaled,
# age decides who is a neighbour. A tree would not care (it asks one column at a time); a neighbour model and a
# regularised logistic regression do.
#
# ## 5 · One column transformer (6')
#
# ✏️ **Exercise** · Write `screen_pipeline()`: a `make_pipeline` of a `ColumnTransformer` and
# `LogisticRegression()`. The transformer sends `U.FEATURES` to a `StandardScaler`, `["bmi"]` to
# `make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())` and `["sector"]` to a
# `OneHotEncoder(handle_unknown="ignore")`.

# %%
def screen_pipeline():
    # BEGIN SOLUTION
    pre = ColumnTransformer([
        ("numbers", StandardScaler(), U.FEATURES),
        ("bmi", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ["bmi"]),
        ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    return make_pipeline(pre, LogisticRegression())
    # END SOLUTION


# %%
cols = U.FEATURES + ["bmi", "sector"]
try:
    pipe = screen_pipeline().fit(train[cols], train.fell_6m)
    pipe_auc = float(roc_auc_score(val.fell_6m, pipe.predict_proba(val[cols])[:, 1]))
    n_inputs = int(pipe[0].transform(train[cols].head(1)).shape[1])
except NotImplementedError:
    pipe_auc, n_inputs = U.reference("d3_2_pipe")
U.keep("d3_2_pipe", [pipe_auc, n_inputs])
print(f"one pipeline, {n_inputs} model inputs from {len(cols)} columns; validation AUC {pipe_auc:.3f}")
U.check(n_inputs == len(U.FEATURES) + 2 + 4, "nine numbers, BMI and its flag, four sector columns")
U.save("D3_2_preprocessing.md", f"# Lab 3.2\n\nNeighbour model AUC raw {raw_auc:.3f}, scaled {scaled_auc:.3f}.\n\n"
       f"Pipeline with {n_inputs} inputs: validation AUC {pipe_auc:.3f}.\n")

# %% [markdown]
# Every step inside the pipeline is fitted when the pipeline is fitted, on the rows it is given, and only applied
# to the rows it scores. That is the whole point of Session 3.
#
# ## ✱ Stretch · Who is missing?
#
# Compute the share of missing BMI for clients who live alone and for the others in the training rows. What
# would a mean imputation without a flag hide?

# %% [markdown]
# ## Checkpoint
#
# 1. Why fit the imputer on training rows and not on all rows? <details><summary>Answer</summary>The median of
#    all rows includes validation and test clients: their information would leak into the model.</details>
# 2. A sector opens after training. What does the encoder do with it? <details><summary>Answer</summary>With
#    `handle_unknown="ignore"` it gives all zeros, so the model treats the client like an unspecified sector; it
#    should be noticed and reviewed (Lab 4.2).</details>
# 3. Does a decision tree need scaled columns? <details><summary>Answer</summary>No: it splits one column at a
#    time on a threshold, which a change of unit does not affect.</details>
#
# **Carry forward.** `artifacts/D3_2_preprocessing.md`. Lab 3.3 starts from a supplied pilot file.
