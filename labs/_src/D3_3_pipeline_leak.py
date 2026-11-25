# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.3 · Selection before the split
#
# Day 3 · Session 3 (Pipelines fitted on training data only) · 40 min.
#
# Hurbuth piloted home activity sensors with 240 clients: 400 columns of daily activity summaries per client.
# A vendor's report says a model on the 20 best columns predicts falls well. You reproduce the report, then
# move one line of code and watch the result disappear. (In this synthetic pilot the 400 columns are random
# numbers: nothing in them predicts a fall.)
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/sensor_pilot.csv` and `data/assessments.csv`, supplied. Nothing from Lab 3.2.
#
# **Tasks**
# - [ ] 1 · Meet the pilot: 240 clients and 400 activity columns (4')
# - [ ] 2 · Select the 20 best columns on all rows, then cross-validate (8')
# - [ ] 3 · Do the same with the selection inside a pipeline (8')
# - [ ] 4 · Shuffle the labels and repeat both (8')
# - [ ] 5 · Scaler and imputer fitted on everything: measure the difference (7')
# - [ ] 6 · Write the rule you will follow (5')
# - [ ] ✱ Stretch · Repeat with 5 and with 100 selected columns

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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

pilot = U.sensor_pilot()
X = pilot.filter(like="activity_")
y = pilot.fell_6m
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=U.SEED)

# %% [markdown]
# ## 1 · The pilot (4')

# %%
print(f"{len(pilot)} clients, {X.shape[1]} activity columns, {int(y.sum())} fell within six months")
print(X.iloc[:3, :6].to_string())

# %% [markdown]
# More columns than clients: with 400 columns and 240 rows, some columns will match the label by luck alone.
#
# ## 2 · Select first, then cross-validate (8')
#
# This is what the vendor's notebook did: keep the 20 columns most related to the label (`SelectKBest` with
# `f_classif`), using **all 240 clients**, then cross-validate a logistic regression on those 20 columns.
#
# ✏️ **Exercise** · Write `select_then_cv(X, y, k)`: fit `SelectKBest(f_classif, k=k)` on all of `X` and `y`,
# transform `X`, and return the mean AUC of `cross_val_score(LogisticRegression(max_iter=1000), ..., cv=CV,
# scoring="roc_auc")`.

# %%
def select_then_cv(X, y, k: int = 20) -> float:
    # BEGIN SOLUTION
    chosen = SelectKBest(f_classif, k=k).fit_transform(X, y)
    return float(cross_val_score(LogisticRegression(max_iter=1000), chosen, y, cv=CV, scoring="roc_auc").mean())
    # END SOLUTION


# %%
try:
    leaky = select_then_cv(X, y)
except NotImplementedError:
    leaky = U.reference("d3_3_leaky")
U.keep("d3_3_leaky", leaky)
print(f"selection on all rows, then five-fold AUC: {leaky:.3f}")
U.check(leaky > 0.7, "the leaky procedure reports a strong model")

# %% [markdown]
# ## 3 · The selection inside the pipeline (8')
#
# ✏️ **Exercise** · Write `pipeline_cv(X, y, k)`: the mean five-fold AUC of
# `make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000))`. Now each fold chooses its 20
# columns from its own four fifths.

# %%
def pipeline_cv(X, y, k: int = 20) -> float:
    # BEGIN SOLUTION
    pipe = make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000))
    return float(cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc").mean())
    # END SOLUTION


# %%
try:
    honest = pipeline_cv(X, y)
except NotImplementedError:
    honest = U.reference("d3_3_honest")
U.keep("d3_3_honest", honest)
print(f"selection inside each fold: {honest:.3f}  (a coin is 0.5)")
U.check(abs(honest - 0.5) < 0.1, "the honest procedure finds nothing, as it should on random columns")

# %% [markdown]
# 🔍 **Look at** the two numbers. Same data, same model, same folds. The only change is *where* the selection
# learnt: on all rows (so it had seen the labels of every scored fold) or inside each fold.
#
# ## 4 · Shuffled labels (8')
#
# A useful test for any pipeline: shuffle the labels, so no column can predict them, and run it again. An honest
# procedure drops to about 0.5. A leaky one keeps finding "signal".

# %%
rng = np.random.default_rng(U.SEED)
y_shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
try:
    shuffled = (select_then_cv(X, y_shuffled), pipeline_cv(X, y_shuffled))
except NotImplementedError:
    shuffled = tuple(U.reference("d3_3_shuffled"))
U.keep("d3_3_shuffled", list(shuffled))
print(f"shuffled labels: selection first {shuffled[0]:.3f}, inside the pipeline {shuffled[1]:.3f}")
U.check(shuffled[0] > 0.7 and abs(shuffled[1] - 0.5) < 0.1, "the leaky procedure scores well even on shuffled labels")

# %% [markdown]
# ## 5 · Scaler and imputer fitted on everything (7')
#
# Not every leak is that large. On the main assessments, fit `StandardScaler` and a median `SimpleImputer` on
# **all** rows (training, validation and test), then train a logistic regression on the training rows and score
# validation. Compare with the same steps fitted on the training rows only.
#
# ✏️ **Exercise** · Write `auc_with_prep(fit_rows)`: fit `make_pipeline(SimpleImputer(strategy="median"),
# StandardScaler())` on `fit_rows[cols]`, transform the training and validation rows with it, fit
# `LogisticRegression()` on the transformed training rows and return the validation AUC.

# %%
df = U.assessments()
train, val, _ = U.split(df)
cols = U.FEATURES + ["bmi"]


def auc_with_prep(fit_rows: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(fit_rows[cols])
    model = LogisticRegression().fit(prep.transform(train[cols]), train.fell_6m)
    return float(roc_auc_score(val.fell_6m, model.predict_proba(prep.transform(val[cols]))[:, 1]))
    # END SOLUTION


# %%
try:
    small = (auc_with_prep(df), auc_with_prep(train))
except NotImplementedError:
    small = tuple(U.reference("d3_3_small"))
U.keep("d3_3_small", list(small))
print(f"validation AUC, preparation fitted on all rows {small[0]:.4f}, on training rows {small[1]:.4f}; "
      f"difference {small[0] - small[1]:+.4f}")
U.save("D3_3_pipeline_leak.md", "# Lab 3.3\n\n| procedure | real labels | shuffled labels |\n|---|---|---|\n"
       f"| selection on all rows, then CV | {leaky:.3f} | {shuffled[0]:.3f} |\n"
       f"| selection inside the pipeline | {honest:.3f} | {shuffled[1]:.3f} |\n\n"
       f"Scaler and imputer on all rows: {small[0]:.4f}; on training rows: {small[1]:.4f}.\n")

# %% [markdown]
# Here the difference is tiny: a mean and a spread of nine columns carry little about the labels. It would not
# stay tiny with a small dataset, a target-based encoding or a step that looks at the labels. Do not rely on the
# size; follow the rule.
#
# ## 6 · The rule (5')
#
# Write in one or two sentences the rule you will follow from now on for every step that learns from data
# (selection, scaling, imputing, encoding), and how a pipeline enforces it in cross-validation.

# %% [markdown]
# *Your rule here.*

# %% [markdown]
# ## ✱ Stretch · How many columns?
#
# Repeat tasks 2 and 3 with `k = 5` and `k = 100`. Which `k` makes the leaky number most impressive, and why?

# %% [markdown]
# ## Checkpoint
#
# 1. A notebook scales all rows, then splits. Is the test score invalid? <details><summary>Answer</summary>It is
#    leaked, if only a little: the scaler saw the test rows. Fix it by fitting the scaler on training rows,
#    which a pipeline does for you.</details>
# 2. Why does shuffling the labels expose the selection leak? <details><summary>Answer</summary>With shuffled
#    labels nothing can be predicted, so any score well above 0.5 comes from the procedure, not the data.
#    </details>
# 3. What does `cross_val_score(pipeline, X, y)` refit in every fold? <details><summary>Answer</summary>Every
#    step of the pipeline: selection, scaling, imputing and the model, on that fold's training part only.
#    </details>
#
# **Carry forward.** `artifacts/D3_3_pipeline_leak.md`. Lab 3.4 starts from a supplied export.
