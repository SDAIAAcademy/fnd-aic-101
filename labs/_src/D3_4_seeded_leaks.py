# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.4 · Three seeded leaks
#
# Day 3 · Session 4 (Target, duplicate and temporal leakage) · 40 min. **Graded lab.**
#
# A former analyst left a working pipeline and a test AUC that impressed everyone. It runs on an export of the
# assessment system, `data/leaky/assessments_export.csv`, described in `data/leaky/columns.md`. Three leaks were
# seeded in it: a field recorded after the outcome, copies across the split, and information from the future.
# Find each one, show the evidence, fix it and rescore.
#
# **AI policy: AI-off** for the search and your first attempt; AI-tutor for debugging after. The leak log is
# yours.
#
# **Reference state.** The export and its column notes, supplied. Nothing from Lab 3.3.
#
# **What is graded.** The leak log: each leak named, its evidence (a number from your run) and its fix, and the
# score before and after.
#
# **Tasks**
# - [ ] 1 · Run the working pipeline and note its score (5')
# - [ ] 2 · Look for a field recorded after the outcome (8')
# - [ ] 3 · Look for copies across the split (8')
# - [ ] 4 · Look for information from the future (8')
# - [ ] 5 · Fix all three and rescore (7')
# - [ ] 6 · Write the leak log: each leak, its evidence, its fix (4')
# - [ ] ✱ Stretch · Fix the leaks one at a time and rank them by how much each moved the score

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

export = U.leaky_export()
print((U.DATA / "leaky" / "columns.md").read_text(encoding="utf-8"))

# %% [markdown]
# ## 1 · The working pipeline (5')
#
# This is the analyst's code, unchanged. Run it and write down the test AUC.

# %%
ANALYST_FEATURES = U.FEATURES + ["physio_referral", "client_falls_total"]


def analyst_pipeline(data: pd.DataFrame, features: list[str]) -> float:
    tr, te = train_test_split(data, test_size=0.3, random_state=1)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[features], tr.fell_6m)
    return float(roc_auc_score(te.fell_6m, model.predict_proba(te[features])[:, 1]))


before = analyst_pipeline(export, ANALYST_FEATURES)
print(f"the analyst's test AUC: {before:.3f}")

# %% [markdown]
# ## 2 · A field recorded after the outcome (8')
#
# A **target leak** is a feature that is a consequence of the label. One way to find it: score each feature on
# its own. A single feature that ranks fallers almost perfectly deserves a question: when is it recorded?
#
# ✏️ **Exercise** · Write `single_feature_auc(data, features)`: a Series, one AUC per feature, of the feature's
# raw value as a score against `fell_6m`, sorted from highest to lowest.

# %%
def single_feature_auc(data: pd.DataFrame, features: list[str]) -> pd.Series:
    # BEGIN SOLUTION
    return pd.Series({f: roc_auc_score(data.fell_6m, data[f]) for f in features}).sort_values(ascending=False)
    # END SOLUTION


# %%
try:
    alone = single_feature_auc(export, ANALYST_FEATURES)
except NotImplementedError:
    alone = U.reference("d3_4_alone")
U.keep("d3_4_alone", alone)
print(alone.round(3).to_string())

# %% [markdown]
# Read the column notes for the top feature. Is it known on the day of the assessment?
#
# ## 3 · Copies across the split (8')
#
# A **duplicate leak**: the same assessment on both sides of the split. The model has seen the answer for an
# identical row.
#
# ✏️ **Exercise** · Write `copies_across(tr, te)`: the number of rows of `te` whose `assessment_id` also appears
# in `tr`. Then the cell below measures it on the analyst's own split.

# %%
def copies_across(tr: pd.DataFrame, te: pd.DataFrame) -> int:
    # BEGIN SOLUTION
    return int(te.assessment_id.isin(set(tr.assessment_id)).sum())
    # END SOLUTION


# %%
tr_a, te_a = train_test_split(export, test_size=0.3, random_state=1)
dupes = int(export.duplicated().sum())
try:
    across = copies_across(tr_a, te_a)
except NotImplementedError:
    across = U.reference("d3_4_across")
U.keep("d3_4_across", across)
print(f"exact duplicate rows in the export: {dupes}; test rows with a copy in training: {across} of {len(te_a)}")
U.check(across > 0, "some test assessments were also in training")

# %% [markdown]
# ## 4 · Information from the future (8')
#
# A **temporal leak**: a feature, or the split itself, uses what happened after the prediction date. Two checks:
#
# - Does a feature of a row depend on later rows? Compare `client_falls_total` with the falls recorded on or
#   before the row's own date.
# - Does the split put later assessments in training and earlier ones in test?
#
# ✏️ **Exercise** · Write `future_share(data)`: the share of rows whose `client_falls_total` is larger than the
# sum of `prior_falls_12m` over that client's assessments dated on or before the row (use the export without its
# duplicate rows).

# %%
def future_share(data: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    d = data.drop_duplicates().sort_values(["client_id", "assessment_date"])
    so_far = d.groupby("client_id").prior_falls_12m.cumsum()
    return float((d.client_falls_total > so_far).mean())
    # END SOLUTION


# %%
try:
    future = future_share(export)
except NotImplementedError:
    future = U.reference("d3_4_future")
U.keep("d3_4_future", future)
later_in_train = float((tr_a.assessment_date >= U.VAL_END).mean())
print(f"rows whose client_falls_total counts later assessments: {future:.1%}")
print(f"training rows from 2025, the most recent period: {later_in_train:.1%}")
U.check(future > 0.1, "client_falls_total looks ahead for a good share of rows")

# %% [markdown]
# ## 5 · Fix all three and rescore (7')
#
# ✏️ **Exercise** · Write `fixed_pipeline(data)`: drop the duplicate rows, keep only `U.FEATURES`, split by date
# with `U.split`, fit the same model on the training period and return the AUC on the test period.

# %%
def fixed_pipeline(data: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    clean = data.drop_duplicates(subset="assessment_id")
    tr, _, te = U.split(clean)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[U.FEATURES], tr.fell_6m)
    return float(roc_auc_score(te.fell_6m, model.predict_proba(te[U.FEATURES])[:, 1]))
    # END SOLUTION


# %%
try:
    after = fixed_pipeline(export)
except NotImplementedError:
    after = U.reference("d3_4_after")
U.keep("d3_4_after", after)
print(f"test AUC before {before:.3f}, after {after:.3f}")
U.check(after < before - 0.1, "the honest score is far below the analyst's")

# %% [markdown]
# The honest number is lower and it is the useful one: it says what the screen will do for clients assessed
# next year. The analyst's number described a model that will not exist in use, because the referral is not
# known on assessment day, the copies will not be there and the future totals cannot be computed.
#
# ## 6 · The leak log (4')
#
# Complete the table in the cell below with your numbers (evidence) and your fix, then run it to save the log.
# The analyst who fixes the pipeline does not sign it off: the reviewer is the quality and data protection
# officer, and the clinical director accepts the new score.

# %%
log = pd.DataFrame([
    {"leak": "target", "column or step": "physio_referral",
     "evidence": f"AUC alone {alone.get('physio_referral', float('nan')):.3f}; recorded during the follow-up",
     "fix": "drop the column"},
    {"leak": "duplicate", "column or step": "the random split of the export",
     "evidence": f"{across} test rows also in training",
     "fix": "drop duplicates by assessment_id before splitting"},
    {"leak": "temporal", "column or step": "client_falls_total and the random split",
     "evidence": f"{future:.1%} of rows count later assessments; {later_in_train:.1%} of training rows come from 2025",
     "fix": "drop the column (or count only earlier assessments) and split by date"},
])
U.save("D3_4_leak_log.md", f"# Lab 3.4 · leak log\n\nTest AUC before {before:.3f}, after {after:.3f}. "
       "Prepared by the analyst; reviewed by the quality and data protection officer; accepted by the clinical director.\n\n"
       + log.to_markdown(index=False) + "\n")
print(log.to_string(index=False))

# %% [markdown]
# ## ✱ Stretch · One fix at a time
#
# Starting from the analyst's pipeline, apply one fix at a time (drop the referral; deduplicate; drop the total
# and split by date) and record the AUC after each. Which leak did the most damage here? Would that order hold for
# another dataset?

# %% [markdown]
# ## Checkpoint
#
# 1. A feature alone gives AUC 0.90 when the best clinical feature gives 0.69. What is your first question?
#    <details><summary>Answer</summary>When is it recorded? A feature that good is often a consequence of the
#    outcome.</details>
# 2. Why do duplicates inflate a score even when the model is simple? <details><summary>Answer</summary>A copy in
#    training means the model has been fitted on the very row it is scored on; flexible models exploit it most,
#    but every model gains a little.</details>
# 3. The screen will be used on next year's clients. Which split matches that? <details><summary>Answer</summary>A
#    split by date: train on the past, test on the most recent period.</details>
#
# **Carry forward.** `artifacts/D3_4_leak_log.md` (graded). Day 4 starts from the clean assessments.
