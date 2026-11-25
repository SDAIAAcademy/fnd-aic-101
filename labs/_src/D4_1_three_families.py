# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.1 · Three families and a baseline
#
# Day 4 · Session 1 (Trees, linear models and small neural networks) · 40 min.
#
# The workflow of the whole course in one notebook: one preprocessing pipeline, a baseline, a logistic
# regression, a decision tree and a small neural network, all fitted on the training period with a fixed seed,
# compared on validation, and the chosen one scored once on test. It must run end to end from a fresh start.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/assessments.csv` and the fixed split. Nothing from Day 3's labs.
#
# **Tasks**
# - [ ] 1 · One preprocessing pipeline for every model (6')
# - [ ] 2 · Fit a baseline, a logistic regression, a tree and a small neural network (10')
# - [ ] 3 · The validation table with a fixed seed (8')
# - [ ] 4 · Run again, then with another seed: what changes (6')
# - [ ] 5 · Choose, score the test set once and save the table (10')
# - [ ] ✱ Stretch · Change the size of the neural network's hidden layer and watch the validation score

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
import warnings
import numpy as np
import pandas as pd
import hurbuth as U
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
train, val, test = U.split()
COLS = U.FEATURES + ["bmi", "sector"]
print(f"training {len(train)}, validation {len(val)}, test {len(test)}; columns {COLS}")

# %% [markdown]
# ## 1 · One preprocessing step for all (6')
#
# Every model gets exactly the same inputs, so the comparison is about the families and nothing else.
#
# ✏️ **Exercise** · Write `preprocess()`: a new `ColumnTransformer` that scales `U.FEATURES`, imputes and scales
# `bmi` with a missing flag, and one-hot encodes `sector` (as in Lab 3.2).

# %%
def preprocess() -> ColumnTransformer:
    # BEGIN SOLUTION
    return ColumnTransformer([
        ("numbers", StandardScaler(), U.FEATURES),
        ("bmi", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ["bmi"]),
        ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    # END SOLUTION


# %%
def safe_preprocess() -> ColumnTransformer:
    try:
        return preprocess()
    except NotImplementedError:
        print("ℹ️  using the reference preprocessing: finish the exercise above to use your own")
        U.check(not U.STRICT, "the solution built its own preprocessing")
        return ColumnTransformer([
            ("numbers", StandardScaler(), U.FEATURES),
            ("bmi", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ["bmi"]),
            ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
        ])


print(safe_preprocess())

# %% [markdown]
# ## 2 · Four candidates (10')
#
# - **Baseline**: `DummyClassifier(strategy="prior")` gives every client the training base rate as a score.
# - **Logistic regression**: a weighted sum of the inputs, squeezed into 0 to 1.
# - **Decision tree**: `max_depth=4`, a few yes-or-no questions.
# - **Small neural network**: `MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000,
#   random_state=seed)`: sixteen units that each compute a weighted sum and bend it, then one more weighted sum.
#   It is fitted by gradient steps on a loss, as on Day 1, starting from random weights.
#
# ✏️ **Exercise** · Write `candidates(seed)`: a dictionary of the four models, each a pipeline that starts with
# `safe_preprocess()`.

# %%
def candidates(seed: int) -> dict:
    # BEGIN SOLUTION
    return {
        "baseline": make_pipeline(safe_preprocess(), DummyClassifier(strategy="prior")),
        "logistic regression": make_pipeline(safe_preprocess(), LogisticRegression()),
        "decision tree": make_pipeline(safe_preprocess(), DecisionTreeClassifier(max_depth=4, random_state=seed)),
        "small neural network": make_pipeline(safe_preprocess(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0,
                                                                               max_iter=2000, random_state=seed)),
    }
    # END SOLUTION


# %% [markdown]
# ## 3 · The validation table (8')

# %%
def validation_table(models: dict) -> pd.DataFrame:
    rows = {}
    for name, m in models.items():
        m.fit(train[COLS], train.fell_6m)
        s = m.predict_proba(val[COLS])[:, 1]
        rows[name] = {"validation AUC": roc_auc_score(val.fell_6m, s), "validation accuracy": accuracy_score(val.fell_6m, s >= 0.5)}
    return pd.DataFrame(rows).T


try:
    models = candidates(U.SEED)
    table = validation_table(models)
except NotImplementedError:
    models, table = None, U.reference("d4_1_table")
U.keep("d4_1_table", table)
print(table.round(3).to_string())
U.check(abs(table.loc["baseline", "validation AUC"] - 0.5) < 1e-9, "the baseline ranks like a coin")

# %% [markdown]
# ## 4 · Same seed, other seed (6')
#
# Run the table again with the same seed, then with seed 7. Which rows change?

# %%
try:
    again = validation_table(candidates(U.SEED))
    other = validation_table(candidates(7))
except NotImplementedError:
    again, other = U.reference("d4_1_again"), U.reference("d4_1_other")
U.keep("d4_1_again", again)
U.keep("d4_1_other", other)
seeds = pd.DataFrame({"seed 101": table["validation AUC"], "seed 101 again": again["validation AUC"],
                      "seed 7": other["validation AUC"]})
print(seeds.round(4).to_string())
U.check((seeds["seed 101"] - seeds["seed 101 again"]).abs().max() < 1e-12, "the same seed gives the same numbers")

# %% [markdown]
# 🔍 **Look at** the neural network's row. Its starting weights are random, so another seed gives a slightly
# different model. A fixed seed makes a run repeatable; it does not make the result less dependent on luck. When
# two models differ by less than the seed-to-seed change, they are tied.
#
# ## 5 · Choose, test once, save (10')
#
# ✏️ **Exercise** · Write `choose(table)`: the name of the model (not the baseline) with the highest validation
# AUC; but if a simpler model is within 0.005 of it, choose the simpler one (simplest first: logistic regression,
# decision tree, small neural network).

# %%
SIMPLEST_FIRST = ["logistic regression", "decision tree", "small neural network"]


def choose(table: pd.DataFrame) -> str:
    # BEGIN SOLUTION
    best = table.loc[SIMPLEST_FIRST, "validation AUC"].max()
    return next(n for n in SIMPLEST_FIRST if table.loc[n, "validation AUC"] >= best - 0.005)
    # END SOLUTION


# %%
try:
    chosen = choose(table)
except NotImplementedError:
    chosen = U.reference("d4_1_chosen")
U.keep("d4_1_chosen", chosen)
final = candidates(U.SEED)[chosen] if models is not None else make_pipeline(safe_preprocess(), LogisticRegression())
final.fit(train[COLS], train.fell_6m)
test_auc = float(roc_auc_score(test.fell_6m, final.predict_proba(test[COLS])[:, 1]))
print(f"chosen on validation: {chosen}; test AUC, opened once: {test_auc:.3f}")
results = table.assign(**{"test AUC": [test_auc if n == chosen else np.nan for n in table.index]})
U.save("D4_1_families.md", "# Lab 4.1 · four candidates, seed 101\n\n" + results.round(3).to_markdown()
       + f"\n\nSeed check:\n\n" + seeds.round(4).to_markdown() + f"\n\nChosen: {chosen}.\n")

# %% [markdown]
# ## ✱ Stretch · The network's size
#
# Try `hidden_layer_sizes` of `(4,)`, `(64,)` and `(64, 64)`, and `alpha` of 0.01. Which setting overfits? Do you
# get a model clearly better than the logistic regression, or only a different one?

# %% [markdown]
# ## Checkpoint
#
# 1. Why give every model the same preprocessing? <details><summary>Answer</summary>So the comparison measures the
#    families, not differences in their inputs.</details>
# 2. The network beats the logistic regression by 0.003 AUC with one seed and loses with another. What do you
#    report? <details><summary>Answer</summary>That they are tied, and choose the simpler model, which is easier to
#    explain and check.</details>
# 3. What would make this notebook fail to reproduce on a colleague's laptop? <details><summary>Answer</summary>A
#    missing seed, a hidden cell run out of order, a file that is not in the repository, or different library
#    versions (the lock file pins them).</details>
#
# **Carry forward.** `artifacts/D4_1_families.md`. Lab 4.2 refits the standard screen itself.
