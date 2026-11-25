# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.4 · The capstone starter
#
# Day 4 · Session 4 (Where language models fit, and the capstone) · 35 min.
#
# The capstone applies the week to a new screen: unplanned hospital admission within 90 days. This notebook is
# your starting point. It runs end to end, but it is not yet honest: like the analyst's pipeline in Lab 3.4, it
# takes every column it is given. You run it, check its data, and plan your report.
#
# **AI policy: AI-off** for the leak check and the plan; AI-full is allowed for the capstone's code after today,
# and you explain every line in the defence.
#
# **Reference state.** `capstone/admissions.csv` and `capstone/README.md` (the brief, the data dictionary, the
# costs and the rubric). Nothing from the earlier labs.
#
# **Tasks**
# - [ ] 1 · Read the capstone brief and the data dictionary (5')
# - [ ] 2 · Run the starter end to end with the fixed seed (8')
# - [ ] 3 · Check the data with the leak checklist of Lab 3.4 (7')
# - [ ] 4 · Plan the report: baseline, two models, threshold, model card (10')
# - [ ] 5 · Create the repository and commit the starter (5')
# - [ ] ✱ Stretch · Work out the cost ratio of the admission screen and the threshold it implies

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
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# %% [markdown]
# ## 1 · The brief (5')
#
# Read `capstone/README.md`: what the screen is for, the columns and when each is recorded, the costs, what you
# submit and the rubric.

# %%
print((U.LABS / "capstone" / "README.md").read_text(encoding="utf-8")[:1800])

# %% [markdown]
# ## 2 · The starter, end to end (8')
#
# Run every cell from the top. The split is the course's split by date; the seed is fixed.

# %%
SEED = 101
TARGET = "admitted_90d"
data = U.admissions()
train, val, test = U.split(data)
print(f"training {len(train)}, validation {len(val)}, test {len(test)}; base rate in training {train[TARGET].mean():.3f}")

STARTER_NUMBERS = ["age", "lives_alone", "n_medications", "er_visits_12m", "heart_failure", "diabetes",
                   "uses_walking_aid", "discharge_letter_received"]


def starter_models(numbers: list[str]) -> dict:
    prep = lambda: ColumnTransformer([
        ("numbers", StandardScaler(), numbers),
        ("bmi", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ["bmi"]),
        ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    return {"baseline": make_pipeline(prep(), DummyClassifier(strategy="prior")),
            "logistic regression": make_pipeline(prep(), LogisticRegression()),
            "decision tree": make_pipeline(prep(), DecisionTreeClassifier(max_depth=4, random_state=SEED))}


def validation_auc(numbers: list[str]) -> pd.Series:
    cols = numbers + ["bmi", "sector"]
    out = {}
    for name, m in starter_models(numbers).items():
        m.fit(train[cols], train[TARGET])
        out[name] = roc_auc_score(val[TARGET], m.predict_proba(val[cols])[:, 1])
    return pd.Series(out, name="validation AUC")


starter = validation_auc(STARTER_NUMBERS)
print(starter.round(3).to_string())

# %% [markdown]
# 🔍 **Look at** the logistic regression's score. For a screen built from a home visit, is that believable?
#
# ## 3 · The leak checklist (7')
#
# From Lab 3.4: (1) score each column alone and ask when the best ones are recorded; (2) look for copies across the
# split; (3) look for anything from after the assessment date, in a column or in the split.
#
# ✏️ **Exercise** · Write `leak_check(df)`: a DataFrame with one row per column in `STARTER_NUMBERS`, its AUC alone
# against `admitted_90d` on the training rows (`alone_auc`), sorted from highest; then return also the number of
# `record_id` values that appear more than once in `df`.

# %%
def leak_check(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # BEGIN SOLUTION
    tr = df[df.assessment_date < U.TRAIN_END]
    alone = pd.DataFrame({"alone_auc": {c: roc_auc_score(tr[TARGET], tr[c]) for c in STARTER_NUMBERS}})
    return alone.sort_values("alone_auc", ascending=False), int(df.record_id.duplicated().sum())
    # END SOLUTION


# %%
try:
    alone, repeated = leak_check(data)
except NotImplementedError:
    alone, repeated = U.reference("d4_4_alone"), U.reference("d4_4_repeated")
U.keep("d4_4_alone", alone)
U.keep("d4_4_repeated", repeated)
print(alone.round(3).to_string())
print(f"record_id values that appear more than once: {repeated}")

# %% [markdown]
# Read the data dictionary for the top column. Then decide which columns stay.
#
# ✏️ **Exercise** · Write `honest_numbers()`: the list of numeric columns your report will use.

# %%
def honest_numbers() -> list[str]:
    # BEGIN SOLUTION
    return [c for c in STARTER_NUMBERS if c != "discharge_letter_received"]
    # END SOLUTION


# %%
try:
    numbers = honest_numbers()
except NotImplementedError:
    numbers = None
if numbers is not None:
    honest = validation_auc(numbers)
else:
    honest = U.reference("d4_4_honest")
U.keep("d4_4_honest", honest)
print(pd.DataFrame({"starter": starter, "without the leak": honest}).round(3).to_string())
U.check(honest["logistic regression"] < starter["logistic regression"] - 0.1, "the honest score is much lower")

# %% [markdown]
# ## 4 · Plan the report (10')
#
# Write your plan in the cell below, one line per item. For each, say which cell of your notebook will produce it.
#
# 1. The baseline and the two models you will compare, and why those two.
# 2. The preprocessing, inside a pipeline, and the columns you drop with the reason.
# 3. How you choose the threshold from the costs (eight to one) and the base rate.
# 4. The metrics by sex you will report, with counts, and what you will check about `er_visits_12m`.
# 5. Who owns the model card and who reviews it (roles, two different people).

# %% [markdown]
# *Your plan here.*

# %% [markdown]
# ## 5 · Your repository (5')
#
# Create a public repository on your own GitHub account, add a copy of this notebook and the data file, and make
# the first commit. Keep the history: the defence looks at it. In a terminal:
#
# ```
# git init capstone && cd capstone
# # copy D4_4_capstone_start.ipynb, hurbuth.py and capstone/admissions.csv in
# git add . && git commit -m "Start the admission screen report"
# ```
#
# Commit after each step of your plan, with a message that says what changed.

# %% [markdown]
# ## ✱ Stretch · The cost ratio
#
# With a missed admission costing eight needless visits, the expected cost of flagging a client with probability
# *p* is (1 − *p*) × 1, and of not flagging is *p* × 8. Where are they equal? Compare that threshold with the one
# `U.cheapest_threshold(..., c_fn=8, c_fp=1)` finds on validation, and say why they can differ.

# %% [markdown]
# ## Checkpoint
#
# 1. The starter's validation AUC was high. What made you doubt it? <details><summary>Answer</summary>One column
#    alone ranks almost perfectly, and the data dictionary says it is updated when a letter arrives, after the
#    admission.</details>
# 2. Why keep the repository's history? <details><summary>Answer</summary>The authenticity check inspects it: a
#    report that grew over several commits shows the work was done.</details>
# 3. What does the defence ask you to do? <details><summary>Answer</summary>Re-run the notebook from a fresh start
#    and explain a result the instructor selects.</details>
#
# **Carry forward.** Your capstone repository, with its first commit.
