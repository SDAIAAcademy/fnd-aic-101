# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.3 · Two models by hand, then in code
#
# Day 1 · Session 3 (Nearest neighbours and linear models by hand) · 40 min. **Graded lab.**
#
# Two ways to turn the ten rows into a rule. The first keeps the rows and lets the three most similar clients
# vote. The second throws the rows away and keeps a line. You fit both on paper, then check your paper against
# scikit-learn.
#
# **AI policy: AI-off** for the paper work and your first attempt at the code. AI-tutor for debugging after.
#
# **Reference state.** `data/ten_rows.csv`. Nothing from Labs 1.1 or 1.2.
#
# **What is graded.** Your paper distances and vote for both new clients, your hand slope and intercept, the
# code that reproduces them, and the three sentences of task 5.
#
# **Tasks**
# - [ ] 1 · By hand: distances from two new clients to the ten rows and a three-neighbour vote (10')
# - [ ] 2 · Rescale walk seconds and see whether the vote changes (6')
# - [ ] 3 · By hand: the least-squares line for visit minutes from medicines, from means (10')
# - [ ] 4 · Reproduce both models with scikit-learn and compare with your paper (8')
# - [ ] 5 · Write three sentences on how each model turned examples into a rule (6')
# - [ ] ✱ Stretch · Repeat the vote with k = 1 and k = 5 and say which you trust

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

ten = U.ten_rows()
NEW = pd.DataFrame({"client": ["new client P", "new client Q"], "prior_falls_12m": [1, 0], "walk_seconds": [6.8, 6.0]})
print(ten[["client_id", "prior_falls_12m", "walk_seconds", "fell_6m"]].to_string(index=False))
print()
print(NEW.to_string(index=False))

# %% [markdown]
# ## 1 · Three nearest neighbours, on paper (10')
#
# Treat each client as a point with two coordinates: recorded falls and walk seconds. The distance between two
# clients is the straight-line distance, `sqrt((falls difference)² + (walk difference)²)`.
#
# On paper, for **new client P**: compute the distance to each of the ten rows, circle the three smallest, and
# let those three vote with their `fell_6m`. The majority is the prediction. Do the same for **new client Q**.
#
# ✏️ **Exercise** · Write `distances(rows, falls, walk)`: a Series of the ten distances, indexed like `rows`.
# Then `vote(rows, falls, walk, k)`: the majority `fell_6m` of the `k` nearest rows (1 or 0).

# %%
def distances(rows: pd.DataFrame, falls: float, walk: float) -> pd.Series:
    # BEGIN SOLUTION
    return np.sqrt((rows["prior_falls_12m"] - falls) ** 2 + (rows["walk_seconds"] - walk) ** 2)
    # END SOLUTION


def vote(rows: pd.DataFrame, falls: float, walk: float, k: int = 3) -> int:
    # BEGIN SOLUTION
    nearest = distances(rows, falls, walk).sort_values(kind="stable").index[:k]
    return int(rows.loc[nearest, "fell_6m"].mean() > 0.5)
    # END SOLUTION


# %%
try:
    d_p = distances(ten, 1, 6.8)
    votes = [vote(ten, r.prior_falls_12m, r.walk_seconds) for r in NEW.itertuples()]
except NotImplementedError:
    d_p, votes = U.reference("d1_3_distances"), U.reference("d1_3_votes")
U.keep("d1_3_distances", d_p)
U.keep("d1_3_votes", votes)
print(pd.DataFrame({"client_id": ten.client_id, "distance to P": d_p.round(2), "fell_6m": ten.fell_6m}).to_string(index=False))
print("votes (P, Q):", votes)
U.check(len(votes) == 2 and set(votes) <= {0, 1}, "one 0-or-1 prediction per new client")

# %% [markdown]
# ## 2 · Change the unit, change the neighbours (6')
#
# A nurse could record the walk in tenths of a second (68 in place of 6.8). Nothing about the clients changes.
# Multiply `walk_seconds` by 10 in a copy of the table, do the same for the new clients, and vote again.

# %%
tenths = ten.assign(walk_seconds=ten.walk_seconds * 10)
try:
    votes_tenths = [vote(tenths, r.prior_falls_12m, r.walk_seconds * 10) for r in NEW.itertuples()]
except NotImplementedError:
    votes_tenths = U.reference("d1_3_votes_tenths")
U.keep("d1_3_votes_tenths", votes_tenths)
print("votes in seconds:          ", votes)
print("votes in tenths of a second:", votes_tenths)

# %% [markdown]
# 🔍 **Look at** what decides the distance once the walk is in tenths: a difference of one recorded fall is now
# worth a tenth of a second of walking. The neighbours of a client depend on the units of the columns. On Day 3
# you scale columns before a distance-based model; this is why.
#
# ## 3 · A least-squares line, on paper (10')
#
# The line with the smallest mean squared error has a closed form. With `x` the medicines and `y` the minutes:
#
# - slope `w = Σ (x − mean x)(y − mean y) / Σ (x − mean x)²`
# - intercept `b = mean y − w × mean x`
#
# On paper: compute `mean x` and `mean y`, then a column of `x − mean x`, one of `y − mean y`, their products
# and the squares of the first. Add the columns and divide.
#
# ✏️ **Exercise** · Write `least_squares(x, y)` returning `(w, b)` with these two formulas (no library fit).

# %%
def least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # BEGIN SOLUTION
    dx, dy = x - x.mean(), y - y.mean()
    w = float((dx * dy).sum() / (dx ** 2).sum())
    return w, float(y.mean() - w * x.mean())
    # END SOLUTION


# %%
x = ten.n_medications.to_numpy(dtype=float)
y = ten.visit_minutes.to_numpy(dtype=float)
try:
    w_hand, b_hand = least_squares(x, y)
except NotImplementedError:
    w_hand, b_hand = U.reference("d1_3_line")
U.keep("d1_3_line", [w_hand, b_hand])
print(f"mean x = {x.mean():.1f}, mean y = {y.mean():.1f}")
print(f"w = {w_hand:.3f} minutes per medicine, b = {b_hand:.3f} minutes")

# %% [markdown]
# ## 4 · The same two models in scikit-learn (8')
#
# A scikit-learn model has the same two moves for every family: `fit(X, y)` learns from examples, `predict(X)`
# applies the rule. `X` is a table of features (two dimensions, even with one column), `y` the label.

# %%
knn = KNeighborsClassifier(n_neighbors=3).fit(ten[["prior_falls_12m", "walk_seconds"]], ten.fell_6m)
knn_votes = [int(v) for v in knn.predict(NEW[["prior_falls_12m", "walk_seconds"]])]
line = LinearRegression().fit(ten[["n_medications"]], ten.visit_minutes)
table = pd.DataFrame({"by hand": [votes[0], votes[1], w_hand, b_hand],
                      "scikit-learn": [knn_votes[0], knn_votes[1], float(line.coef_[0]), float(line.intercept_)]},
                     index=["vote for P", "vote for Q", "slope w", "intercept b"])
print(table.round(3).to_string())
U.check(knn_votes == votes, "your votes match KNeighborsClassifier(n_neighbors=3)")
U.check(abs(line.coef_[0] - w_hand) < 1e-9 and abs(line.intercept_ - b_hand) < 1e-9, "your line matches LinearRegression")
U.save("D1_3_by_hand.md", "# Lab 1.3 · by hand and in code\n\n" + table.round(3).to_markdown() + "\n")

# %% [markdown]
# ## 5 · Three sentences (6')
#
# Write in the cell below, in your own words and without an assistant: (1) what the neighbour model keeps from
# the examples and how it predicts; (2) what the line keeps and how it was chosen; (3) one situation in which
# you would trust each one less.

# %% [markdown]
# *Your three sentences here.*

# %% [markdown]
# ## ✱ Stretch · Other values of k
#
# Repeat the vote for P and Q with `k = 1` and `k = 5`. With ten rows, what does `k = 1` trust, and what does
# `k = 9` reduce every prediction to?

# %% [markdown]
# ## Checkpoint
#
# 1. A neighbour model "trains" instantly. Where does its work go? <details><summary>Answer</summary>To prediction
#    time: it keeps every row and measures distances to all of them for each new client.</details>
# 2. Your line predicts a visit length for a client with 30 medicines. Would you trust it? <details><summary>Answer
#    </summary>No. The ten rows run from 1 to 10 medicines; beyond that the line extends a pattern nobody has
#    seen.</details>
# 3. Why did the vote change when the walk moved to tenths? <details><summary>Answer</summary>The distance adds
#    raw differences, so the column with the larger numbers dominates. Scaling puts columns on a common footing.
#    </details>
#
# **Carry forward.** `artifacts/D1_3_by_hand.md` and your paper sheet, which the trainer collects for grading.
# Lab 1.4 starts from `data/assessments.csv`.
