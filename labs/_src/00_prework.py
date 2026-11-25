# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Guided pre-work · a slope and a table
#
# Before Day 1 · 75 min, on your own. Two short parts the course builds on: the slope of a squared error (Day 1,
# Session 2 uses it to fit a line by gradient steps) and the few dataframe operations the labs use (the
# prerequisite Python course does not cover pandas).
#
# **AI policy: AI-tutor.** Ask an assistant to explain, not to answer the exercises. Paste nothing but the public
# examples of this notebook into it.
#
# **Tasks**
# - [ ] 1 · The squared error of one point and its slope (20')
# - [ ] 2 · The mean squared error of several points and its two slopes (20')
# - [ ] 3 · A table: read, select, filter (15')
# - [ ] 4 · A table: group and summarise (15')
# - [ ] 5 · Five self-check questions (5')

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

# %% [markdown]
# ## 1 · One point (20')
#
# A line through the origin predicts *ŷ = w × x*. For one point (*x*, *y*) its squared error is
# *L(w) = (w × x − y)²*. Its slope with respect to *w* is **2 × (w × x − y) × x**: twice the error, times *x*.
#
# Take the point *x* = 3, *y* = 12. At *w* = 2 the prediction is 6, the error is −6, the squared error is 36 and
# the slope is 2 × (−6) × 3 = −36: negative, so increasing *w* lowers the loss.
#
# ✏️ **Exercise** · Write `slope_one(w, x, y)`, then check it against a numerical slope: the change in the loss
# between *w* + 0.001 and *w* − 0.001, divided by 0.002.

# %%
def loss_one(w: float, x: float, y: float) -> float:
    return (w * x - y) ** 2


def slope_one(w: float, x: float, y: float) -> float:
    # BEGIN SOLUTION
    return 2 * (w * x - y) * x
    # END SOLUTION


# %%
try:
    for w in (2.0, 4.0, 5.0):
        numeric = (loss_one(w + 0.001, 3, 12) - loss_one(w - 0.001, 3, 12)) / 0.002
        print(f"w = {w}: your slope {slope_one(w, 3, 12):.3f}, numerical slope {numeric:.3f}")
    U.check(abs(slope_one(2.0, 3, 12) + 36) < 1e-9, "the slope at w = 2 is -36")
except NotImplementedError:
    print("finish slope_one above, then run this cell again")

# %% [markdown]
# At which *w* is the slope zero? That is where the line passes through the point, and the loss is smallest.
#
# ## 2 · Several points, two numbers (20')
#
# With an intercept, the line is *ŷ = w × x + b*, and over *n* points the **mean squared error** is the mean of
# *(w × xᵢ + b − yᵢ)²*. It has two slopes:
#
# - with respect to *w*: **2 × mean(errorᵢ × xᵢ)**;
# - with respect to *b*: **2 × mean(errorᵢ)**.
#
# ✏️ **Exercise** · Write `slopes(w, b, x, y)` returning both, for the four points below.

# %%
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([3.0, 5.0, 6.0, 9.0])


def mse(w, b, x, y):
    return float(np.mean((w * x + b - y) ** 2))


def slopes(w: float, b: float, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # BEGIN SOLUTION
    err = w * x + b - y
    return float(2 * np.mean(err * x)), float(2 * np.mean(err))
    # END SOLUTION


# %%
try:
    g_w, g_b = slopes(1.0, 1.0, x, y)
    print(f"at w = 1, b = 1: mean squared error {mse(1, 1, x, y):.2f}; slopes {g_w:.2f} (w) and {g_b:.2f} (b)")
    U.check(abs(g_w + 13.5) < 1e-9 and abs(g_b + 4.5) < 1e-9, "both slopes are right")
except NotImplementedError:
    print("finish slopes above, then run this cell again")

# %% [markdown]
# Both slopes are negative: raising *w* and raising *b* both lower the error. Day 1 turns this into a loop.
#
# ## 3 · Read, select, filter (15')
#
# The labs keep data in a **dataframe**: a table with named columns. Here are the moves you will see.

# %%
df = U.assessments()
print(df.shape)                       # rows, columns
print(df.columns.tolist())
print(df[["age", "walk_seconds"]].head())  # two columns, first five rows
older = df[df.age >= 85]              # the rows that meet a condition
print(f"{len(older)} assessments of clients aged 85 or more")

# %% [markdown]
# ✏️ **Exercise** · Write `count_alone_with_aid(df)`: the number of assessments where the client lives alone
# **and** uses a walking aid. Combine two conditions with `&`, each in brackets.

# %%
def count_alone_with_aid(df: pd.DataFrame) -> int:
    # BEGIN SOLUTION
    return int(((df.lives_alone == 1) & (df.uses_walking_aid == 1)).sum())
    # END SOLUTION


# %%
try:
    print(f"lives alone and uses a walking aid: {count_alone_with_aid(df)}")
except NotImplementedError:
    print("finish count_alone_with_aid above, then run this cell again")

# %% [markdown]
# ## 4 · Group and summarise (15')

# %%
print(df.sector.value_counts())                       # how many rows per value
print(df.groupby("sector").age.mean().round(1))       # a mean per group

# %% [markdown]
# ✏️ **Exercise** · Write `medicines_by_aid(df)`: the mean of `n_medications` for clients with and without a
# walking aid, as a Series indexed by `uses_walking_aid`.

# %%
def medicines_by_aid(df: pd.DataFrame) -> pd.Series:
    # BEGIN SOLUTION
    return df.groupby("uses_walking_aid").n_medications.mean()
    # END SOLUTION


# %%
try:
    print(medicines_by_aid(df).round(2))
except NotImplementedError:
    print("finish medicines_by_aid above, then run this cell again")

# %% [markdown]
# ## 5 · Self-check (5')
#
# 1. At the point (3, 12) and *w* = 5, is the slope positive or negative, and which way should *w* move?
#    <details><summary>Answer</summary>Positive (2 × 3 × 3 = 18): the prediction is too high, so *w* moves down.
#    </details>
# 2. What does the slope with respect to *b* say when every prediction is too low?
#    <details><summary>Answer</summary>It is negative: raising *b* lowers the error.</details>
# 3. Why must each condition be in brackets when combined with `&`? <details><summary>Answer</summary>`&` binds
#    more tightly than `==` in Python, so without brackets the comparison is read in the wrong order.</details>
# 4. `df.groupby("sector").age.mean()` returns how many numbers? <details><summary>Answer</summary>One per sector.
#    </details>
# 5. What is `df.shape`? <details><summary>Answer</summary>The number of rows and of columns.</details>
