# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.4 · Baseline, line and tree on a fixed split
#
# Day 1 · Session 4 (Training, validation and test sets) · 40 min.
#
# The first honest comparison. The course splits Hurbuth's assessments by date: the oldest train the models,
# the next six months choose between them, the most recent six months are opened once at the end. You compare
# a baseline that predicts no fall for anyone, a logistic regression and a decision tree.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/assessments.csv` and the split in `hurbuth.split()`. Nothing from Labs 1.1 to 1.3.
#
# **Tasks**
# - [ ] 1 · Load the split and check sizes and base rates (5')
# - [ ] 2 · Score a baseline that predicts no fall (5')
# - [ ] 3 · Fit a logistic regression and a tree (8')
# - [ ] 4 · Compare on validation: accuracy and ranking (10')
# - [ ] 5 · Choose on validation, score test once (7')
# - [ ] 6 · Write the results table (5')
# - [ ] ✱ Stretch · Split at random instead of by date and compare

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
import pandas as pd
import hurbuth as U
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = U.assessments()
print(f"{len(df):,} assessments of {df.client_id.nunique():,} clients, "
      f"{df.assessment_date.min():%Y-%m-%d} to {df.assessment_date.max():%Y-%m-%d}")
print("features:", U.FEATURES)

# %% [markdown]
# ## 1 · The fixed split (5')
#
# ✏️ **Exercise** · Write `describe(parts)`: for a dictionary of named DataFrames return a table with one row
# per part and the columns `rows`, `first` (first date), `last` (last date) and `base_rate` (share of
# `fell_6m = 1`).

# %%
train, val, test = U.split(df)


def describe(parts: dict) -> pd.DataFrame:
    # BEGIN SOLUTION
    return pd.DataFrame({name: {"rows": len(p), "first": p.assessment_date.min().date(),
                                "last": p.assessment_date.max().date(), "base_rate": round(p.fell_6m.mean(), 3)}
                         for name, p in parts.items()}).T
    # END SOLUTION


# %%
try:
    sizes = describe({"train": train, "validation": val, "test": test})
except NotImplementedError:
    sizes = U.reference("d1_4_sizes")
U.keep("d1_4_sizes", sizes.astype(str))
print(sizes.to_string())
U.check(len(train) + len(val) + len(test) == len(df), "the three parts hold every row once")
U.check(train.assessment_date.max() < val.assessment_date.min() <= val.assessment_date.max() < test.assessment_date.min(),
        "the parts follow each other in time")

# %% [markdown]
# Why by date? The screen will be used on clients assessed *after* it is built. A validation and a test set that
# come later than the training rows ask the same question: how well does a model built on the past do on the
# months that follow?
#
# ## 2 · A baseline without a model (5')
#
# The cheapest prediction is "nobody falls". Every model must beat it.

# %%
X_tr, y_tr = train[U.FEATURES], train.fell_6m
X_va, y_va = val[U.FEATURES], val.fell_6m
baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
base_acc = accuracy_score(y_va, baseline.predict(X_va))
print(f"baseline accuracy on validation: {base_acc:.3f}")

# %% [markdown]
# 🔍 **Look at** that number. A "model" that never flags anyone is right on most validation rows, because most
# clients do not fall. Accuracy alone will not tell a useful screen from this one.
#
# ## 3 · Two models on the training rows (8')
#
# ✏️ **Exercise** · Write `fit_two(X, y)`: return a dictionary with a fitted `"logistic regression"`
# (`make_pipeline(StandardScaler(), LogisticRegression())`) and a fitted `"decision tree"`
# (`DecisionTreeClassifier(max_depth=4, random_state=U.SEED)`), both fitted on `X` and `y`.

# %%
def fit_two(X, y) -> dict:
    # BEGIN SOLUTION
    return {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y),
            "decision tree": DecisionTreeClassifier(max_depth=4, random_state=U.SEED).fit(X, y)}
    # END SOLUTION


# %%
try:
    models = fit_two(X_tr, y_tr)
except NotImplementedError:
    print("ℹ️  using the reference models: finish the exercise above to fit your own")
    models = {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr),
              "decision tree": DecisionTreeClassifier(max_depth=4, random_state=U.SEED).fit(X_tr, y_tr)}
    U.check(not U.STRICT, "the solution fitted its own models")
models = {"baseline (nobody falls)": baseline, **models}
print(list(models))

# %% [markdown]
# ## 4 · Validation: accuracy and the ranking score (10')
#
# A model gives each client a **score** between 0 and 1 (`predict_proba`). Accuracy turns scores into flags at
# 0.5 and counts right answers. The **ranking score** (the area under the ROC curve, AUC) asks something else:
# pick one client who fell and one who did not, at random; how often does the faller get the higher score?
# 0.5 is a coin, 1.0 is a perfect ranking.
#
# ✏️ **Exercise** · Write `evaluate(models, X, y)`: a table with one row per model and the columns `accuracy`
# and `auc`.

# %%
def evaluate(models: dict, X, y) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = {}
    for name, m in models.items():
        score = m.predict_proba(X)[:, 1]
        rows[name] = {"accuracy": accuracy_score(y, m.predict(X)), "auc": roc_auc_score(y, score)}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    val_table = evaluate(models, X_va, y_va)
except NotImplementedError:
    val_table = U.reference("d1_4_val_table")
U.keep("d1_4_val_table", val_table)
print(val_table.round(3).to_string())
U.check(abs(val_table.loc["baseline (nobody falls)", "auc"] - 0.5) < 1e-9, "the baseline ranks like a coin: AUC 0.5")
U.check(val_table.auc.max() > 0.65, "at least one model ranks clearly better than a coin")

# %% [markdown]
# 🔍 **Look at** the two columns side by side. The accuracies are close to the baseline's; the ranking scores
# are not. The models do separate fallers from non-fallers, but at a cut of 0.5 they flag almost nobody. Where to
# cut is a decision of its own, and it is Day 2's.
#
# ## 5 · Choose on validation, test once (7')
#
# ✏️ **Exercise** · Write `choose(table)`: the name of the model (not the baseline) with the highest validation
# AUC. Then the cell below scores the test set, once, for that model and the baseline.

# %%
def choose(table: pd.DataFrame) -> str:
    # BEGIN SOLUTION
    return str(table.drop(index="baseline (nobody falls)").auc.idxmax())
    # END SOLUTION


# %%
try:
    chosen = choose(val_table)
except NotImplementedError:
    chosen = U.reference("d1_4_chosen")
U.keep("d1_4_chosen", chosen)
X_te, y_te = test[U.FEATURES], test.fell_6m
try:
    test_table = evaluate({k: models[k] for k in ["baseline (nobody falls)", chosen]}, X_te, y_te)
except NotImplementedError:
    test_table = U.reference("d1_4_test_table")
U.keep("d1_4_test_table", test_table)
print("chosen on validation:", chosen)
print(test_table.round(3).to_string())

# %% [markdown]
# ## 6 · The results table (5')

# %%
results = val_table.add_prefix("validation ").join(test_table.add_prefix("test "))
U.save("D1_4_results.md", "# Lab 1.4 · fixed split by date\n\n" + sizes.to_markdown() + "\n\n"
       + results.round(3).to_markdown() + f"\n\nChosen on validation: {chosen}. Test opened once.\n")
print(results.round(3).to_string())

# %% [markdown]
# The empty test cells of the model you did not choose are deliberate: you did not look.
#
# ## ✱ Stretch · A random split
#
# Use `sklearn.model_selection.train_test_split(df, test_size=0.3, random_state=U.SEED)` in place of the split
# by date, refit both models and compare the test AUC with the one above. Write down which of the two numbers
# you would give the clinical director, and why. Day 3 returns to this.

# %% [markdown]
# ## Checkpoint
#
# 1. Why three sets and not two? <details><summary>Answer</summary>Validation is used to choose, so its score is
#    flattering for the winner. The test set took no part in any choice, so its score is a fair estimate.</details>
# 2. The tree's validation accuracy is above the baseline by a point or less. Is the tree useless? <details>
#    <summary>Answer</summary>Not necessarily: its ranking score shows it separates fallers. Accuracy at a cut
#    of 0.5 hides that when only about one client in six falls.</details>
# 3. A colleague scores the test set, changes the tree depth and scores it again. What has the test set become?
#    <details><summary>Answer</summary>A second validation set. A fresh, untouched set is needed for an honest
#    final number.</details>
#
# **Carry forward.** `artifacts/D1_4_results.md`. Lab 2.1 starts from the same fixed split.
