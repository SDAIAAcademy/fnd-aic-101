# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.3 · Four cells by hand, then in code
#
# Day 2 · Session 3 (Confusion matrix, precision and recall) · 35 min.
#
# A fixed fall screen has already scored every validation assessment (a logistic regression on six features,
# fitted on the training period). You turn its scores into flags at a cut-off, count the four cells by hand on
# twenty clients, then compute precision, recall and specificity on all of them and check against scikit-learn.
#
# **AI policy: AI-off** for the hand count and your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/scores/val_scores.csv`, supplied. Nothing from Labs 2.1 or 2.2.
#
# **Tasks**
# - [ ] 1 · Fill the four cells by hand at threshold 0.2 on twenty rows (8')
# - [ ] 2 · Write confusion(y, score, t) (7')
# - [ ] 3 · Precision, recall and specificity on the validation scores (8')
# - [ ] 4 · Accuracy against the base rate (5')
# - [ ] 5 · Check your numbers with scikit-learn (7')
# - [ ] ✱ Stretch · Repeat at threshold 0.15 and say which cells moved

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
from sklearn.metrics import confusion_matrix, precision_score, recall_score

val = U.scores("val")
twenty = val.head(20)
print(f"{len(val)} validation assessments with a score; the first twenty:")
print(twenty[["assessment_id", "score", "fell_6m"]].to_string(index=False))

# %% [markdown]
# ## 1 · Four cells by hand (8')
#
# A client is **flagged** when the score is at least the threshold. On paper, for the twenty rows at 0.2, put
# each row in one cell:
#
# | | fell (1) | did not fall (0) |
# |---|---|---|
# | flagged | true positive (TP): caught | false positive (FP): needless flag |
# | not flagged | false negative (FN): missed | true negative (TN): correct pass |
#
# Write your four counts here (leave `None` if reading ahead).

# %%
MY_TP, MY_FP, MY_FN, MY_TN = None, None, None, None

# %% [markdown]
# ## 2 · The four cells in code (7')
#
# ✏️ **Exercise** · Write `confusion(y, score, t)`: return a dictionary with `tp`, `fp`, `fn` and `tn` for the
# labels `y`, the scores `score` and the threshold `t` (flag when `score >= t`).

# %%
def confusion(y, score, t: float) -> dict:
    # BEGIN SOLUTION
    y = np.asarray(y) == 1
    f = np.asarray(score) >= t
    return {"tp": int((f & y).sum()), "fp": int((f & ~y).sum()), "fn": int((~f & y).sum()), "tn": int((~f & ~y).sum())}
    # END SOLUTION


# %%
try:
    c20 = confusion(twenty.fell_6m, twenty.score, 0.2)
except NotImplementedError:
    c20 = U.reference("d2_3_c20")
U.keep("d2_3_c20", c20)
print("twenty rows at 0.2:", c20)
U.check(sum(c20.values()) == 20, "the four cells hold the twenty rows")
if MY_TP is not None:
    U.check((MY_TP, MY_FP, MY_FN, MY_TN) == (c20["tp"], c20["fp"], c20["fn"], c20["tn"]), "your hand count matches")

# %% [markdown]
# ## 3 · Precision, recall and specificity (8')
#
# - **Recall** (sensitivity): of the clients who fell, the share flagged, `TP / (TP + FN)`.
# - **Precision**: of the clients flagged, the share who fell, `TP / (TP + FP)`.
# - **Specificity**: of the clients who did not fall, the share not flagged, `TN / (TN + FP)`.
#
# ✏️ **Exercise** · Write `rates(c)`: from a dictionary of four cells return `recall`, `precision`,
# `specificity` and `accuracy` (use 0.0 for a precision with no flags at all).

# %%
def rates(c: dict) -> dict:
    # BEGIN SOLUTION
    flagged = c["tp"] + c["fp"]
    return {"recall": c["tp"] / (c["tp"] + c["fn"]),
            "precision": c["tp"] / flagged if flagged else 0.0,
            "specificity": c["tn"] / (c["tn"] + c["fp"]),
            "accuracy": (c["tp"] + c["tn"]) / sum(c.values())}
    # END SOLUTION


# %%
try:
    c_all = confusion(val.fell_6m, val.score, 0.5)
    r_all = rates(c_all)
except NotImplementedError:
    c_all, r_all = U.reference("d2_3_c_all"), U.reference("d2_3_r_all")
U.keep("d2_3_c_all", c_all)
U.keep("d2_3_r_all", r_all)
print("all validation rows at 0.5:", c_all)
print({k: round(v, 3) for k, v in r_all.items()})
U.check(r_all["recall"] < 0.3, "at a cut of 0.5 the screen catches few of the fallers")

# %% [markdown]
# ## 4 · Accuracy against the base rate (5')

# %%
base_rate = float(val.fell_6m.mean())
print(f"base rate {base_rate:.3f}: flagging nobody is right on {1 - base_rate:.3f} of rows; "
      f"the screen at 0.5 is right on {r_all['accuracy']:.3f}")

# %% [markdown]
# 🔍 **Look at** the two accuracies. The screen at 0.5 barely beats doing nothing, yet its recall and precision
# tell you what it really does: of the clients it flags, some share fall; of those who fall, it finds few. The
# cut-off of 0.5 has no special meaning for a rare outcome. Lab 2.4 chooses it from costs.
#
# ## 5 · The same with scikit-learn (7')

# %%
flag = (val.score >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(val.fell_6m, flag).ravel()
sk = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
print("scikit-learn:", sk, "recall", round(recall_score(val.fell_6m, flag), 3),
      "precision", round(precision_score(val.fell_6m, flag, zero_division=0), 3))
U.check(sk == c_all, "your four cells match confusion_matrix")
U.save("D2_3_confusion.md", "# Lab 2.3 · threshold 0.5 on validation\n\n"
       + pd.DataFrame([c_all]).to_markdown(index=False) + "\n\n"
       + pd.DataFrame([{k: round(v, 3) for k, v in r_all.items()}]).to_markdown(index=False) + "\n")

# %% [markdown]
# Note the order: scikit-learn's `confusion_matrix(y, flag)` puts the true label on the rows and starts with
# 0, so `ravel()` gives TN, FP, FN, TP. Mixing up that order is the most common bug in this topic.
#
# ## ✱ Stretch · A lower threshold
#
# Repeat tasks 2 and 3 at 0.15 on all validation rows. Which two cells grew, which two shrank, and which rate rose?

# %% [markdown]
# ## Checkpoint
#
# 1. Precision 0.5 and recall 0.2: describe the screen in words. <details><summary>Answer</summary>Half of the
#    flagged clients fall, but it finds only one faller in five.</details>
# 2. Same screen, same threshold, but a group of clients with twice the base rate. Which rate is most likely to
#    rise: precision or recall? <details><summary>Answer</summary>Precision. With more fallers among the clients,
#    more of the flagged are fallers. Recall is a share of the fallers only, so it does not depend on how many
#    non-fallers there are.</details>
# 3. A colleague reports 88% accuracy. What do you ask first? <details><summary>Answer</summary>The base rate,
#    and the recall and precision at the cut-off used.</details>
#
# **Carry forward.** `artifacts/D2_3_confusion.md`. Lab 2.4 starts from the supplied scores.
