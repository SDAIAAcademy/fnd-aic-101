# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 3.1 · Visit minutes against two baselines
#
# Day 3 · Session 1 (Regression errors against a naive baseline) · 35 min.
#
# The nursing supervisor plans the day's rounds from how long each visit will take. A prediction of minutes is
# a regression: the error is a number of minutes, not a right or wrong. You compute two errors by hand, score
# two naive baselines that need no model, and ask how much a linear model on five features really adds.
#
# **AI policy: AI-off** for the hand errors and your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/assessments.csv` and the fixed split. Nothing from Day 2's labs.
#
# **Tasks**
# - [ ] 1 · Two naive baselines: the training mean and median (5')
# - [ ] 2 · Mean absolute error and root mean squared error by hand on five rows (7')
# - [ ] 3 · A linear model on five features (8')
# - [ ] 4 · Errors on validation against the baselines (8')
# - [ ] 5 · Where the model errs most: residuals by group (7')
# - [ ] ✱ Stretch · Try a decision tree regressor and compare its validation error

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

train, val, _ = U.split()
R = ["n_medications", "uses_walking_aid", "diabetes", "lives_alone", "age"]
print(f"visit minutes in training: mean {train.visit_minutes.mean():.1f}, median {train.visit_minutes.median():.0f}, "
      f"range {train.visit_minutes.min()} to {train.visit_minutes.max()}")

# %% [markdown]
# ## 1 · Two baselines (5')
#
# A regression baseline predicts the same number for every visit: the training **mean** or the training
# **median**. Any model must beat both.
#
# ✏️ **Exercise** · Write `baselines(train)`: return the pair `(mean, median)` of the training visit minutes.

# %%
def baselines(train: pd.DataFrame) -> tuple[float, float]:
    # BEGIN SOLUTION
    return float(train.visit_minutes.mean()), float(train.visit_minutes.median())
    # END SOLUTION


# %%
try:
    MEAN, MEDIAN = baselines(train)
except NotImplementedError:
    MEAN, MEDIAN = U.reference("d3_1_baselines")
U.keep("d3_1_baselines", [MEAN, MEDIAN])
print(f"mean baseline {MEAN:.2f} minutes, median baseline {MEDIAN:.1f} minutes")

# %% [markdown]
# ## 2 · Two errors by hand (7')
#
# For the five validation visits below, the mean baseline predicts the same number each time. On paper write
# each error (prediction minus actual), then:
#
# - **mean absolute error (MAE)**: the average of the errors without their signs, in minutes;
# - **root mean squared error (RMSE)**: the square root of the average squared error, also in minutes. Large
#   errors weigh more in it.

# %%
five = val.head(5)[["assessment_id", "visit_minutes"]].assign(prediction=round(MEAN, 1))
print(five.to_string(index=False))
MY_MAE, MY_RMSE = None, None     # your hand numbers

# %% [markdown]
# ✏️ **Exercise** · Write `mae(y, p)` and `rmse(y, p)` without a library.

# %%
def mae(y, p) -> float:
    # BEGIN SOLUTION
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))
    # END SOLUTION


def rmse(y, p) -> float:
    # BEGIN SOLUTION
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))
    # END SOLUTION


# %%
try:
    hand = (mae(five.visit_minutes, five.prediction), rmse(five.visit_minutes, five.prediction))
except NotImplementedError:
    hand = tuple(U.reference("d3_1_hand"))
U.keep("d3_1_hand", list(hand))
print(f"five rows: MAE {hand[0]:.2f}, RMSE {hand[1]:.2f}")
U.check(hand[1] >= hand[0], "RMSE is never below MAE")
if MY_MAE is not None:
    U.check(abs(MY_MAE - hand[0]) < 0.05 and abs(MY_RMSE - hand[1]) < 0.05, "your hand numbers match")

# %% [markdown]
# ## 3 · A linear model on five features (8')
#
# ✏️ **Exercise** · Write `fit_minutes(train)`: a `LinearRegression` fitted on the columns `R` of the training
# rows with the label `visit_minutes`.

# %%
def fit_minutes(train: pd.DataFrame):
    # BEGIN SOLUTION
    return LinearRegression().fit(train[R], train.visit_minutes)
    # END SOLUTION


# %%
try:
    model = fit_minutes(train)
except NotImplementedError:
    print("ℹ️  using the reference model: finish the exercise above to fit your own")
    model = LinearRegression().fit(train[R], train.visit_minutes)
    U.check(not U.STRICT, "the solution fitted its own model")
coefs = pd.Series(model.coef_, index=R).round(2)
print(coefs.to_string(), f"\nintercept {model.intercept_:.2f}")

# %% [markdown]
# Read each weight as "extra minutes per unit, with the other features held fixed": a walking aid adds about
# its weight in minutes to a visit, each medicine about its own.
#
# ## 4 · Validation errors against the baselines (8')

# %%
y = val.visit_minutes
pred = model.predict(val[R])
table = pd.DataFrame({
    "MAE": [mean_absolute_error(y, np.full(len(y), MEAN)), mean_absolute_error(y, np.full(len(y), MEDIAN)),
            mean_absolute_error(y, pred)],
    "RMSE": [mean_squared_error(y, np.full(len(y), MEAN)) ** 0.5, mean_squared_error(y, np.full(len(y), MEDIAN)) ** 0.5,
             mean_squared_error(y, pred) ** 0.5]}, index=["mean baseline", "median baseline", "linear model"])
print(table.round(2).to_string())
U.check(table.loc["linear model", "MAE"] < 0.7 * table.loc["mean baseline", "MAE"], "the model cuts the baseline's MAE by more than 30%")
try:
    U.check(abs(mae(y, pred) - table.loc["linear model", "MAE"]) < 1e-9, "your mae matches scikit-learn's")
except NotImplementedError:
    pass

# %% [markdown]
# 🔍 **Look at** the median baseline's MAE against the mean's. The median minimises absolute error, the mean
# minimises squared error: each baseline wins on its own measure. Report the model against the baseline that
# matches the error you report.
#
# ## 5 · Where the model errs most (7')
#
# ✏️ **Exercise** · Write `residuals_by(df, pred, column)`: the MAE of the model and the mean residual
# (prediction minus actual) for each value of `column`, with the number of visits.

# %%
def residuals_by(df: pd.DataFrame, pred, column: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    d = df.assign(residual=np.asarray(pred) - df.visit_minutes)
    g = d.groupby(column).residual
    return pd.DataFrame({"visits": g.size(), "MAE": g.apply(lambda r: r.abs().mean()), "mean residual": g.mean()})
    # END SOLUTION


# %%
try:
    by_aid = residuals_by(val, pred, "uses_walking_aid")
    by_sector = residuals_by(val, pred, "sector")
except NotImplementedError:
    by_aid, by_sector = U.reference("d3_1_by_aid"), U.reference("d3_1_by_sector")
U.keep("d3_1_by_aid", by_aid)
U.keep("d3_1_by_sector", by_sector)
print(by_aid.round(2).to_string(), "\n")
print(by_sector.round(2).to_string())
U.save("D3_1_regression.md", "# Lab 3.1 · visit minutes on validation\n\n" + table.round(2).to_markdown() + "\n\n"
       + by_aid.round(2).to_markdown() + "\n\n" + by_sector.round(2).to_markdown() + "\n")

# %% [markdown]
# A mean residual near zero in every group says the model is not biased towards one group; a large MAE in one
# group says it is less precise there. Both matter to a supervisor who plans rounds.
#
# ## ✱ Stretch · A tree regressor
#
# Fit `DecisionTreeRegressor(max_depth=4, random_state=U.SEED)` on the same columns. Does it beat the linear
# model on validation MAE? Why might a line suit this target?

# %% [markdown]
# ## Checkpoint
#
# 1. A model's MAE is 7.9 minutes. Is that good? <details><summary>Answer</summary>It depends on the baseline:
#    here the mean baseline is about 8 minutes, so 7.9 would add almost nothing.</details>
# 2. When does RMSE rise far above MAE? <details><summary>Answer</summary>When a few errors are very large: the
#    squares make them dominate.</details>
# 3. Why report the median baseline next to an MAE? <details><summary>Answer</summary>The median is the best
#    constant for absolute error, so it is the fair naive competitor.</details>
#
# **Carry forward.** `artifacts/D3_1_regression.md`. Lab 3.2 starts from the same fixed split.
