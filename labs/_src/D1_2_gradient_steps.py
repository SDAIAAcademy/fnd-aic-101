# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.2 · Two gradient steps on ten rows
#
# Day 1 · Session 2 (Loss and gradient descent) · 35 min.
#
# The nursing supervisor plans each day from how long visits take. You fit a line that predicts
# `visit_minutes` from `n_medications` on the ten rows: you score a line with the mean squared error, take two
# gradient steps by hand, then let the computer take a few thousand and compare with the exact answer.
#
# **AI policy: AI-off** for the hand work (a calculator is fine). AI-tutor for a Python error after your first
# attempt.
#
# **Reference state.** `data/ten_rows.csv`. Nothing from Lab 1.1.
#
# **Tasks**
# - [ ] 1 · Guess a line for visit minutes and compute its mean squared error by hand (8')
# - [ ] 2 · Write mse(w, b) and check your hand number (5')
# - [ ] 3 · Take two gradient steps by hand from w = 0 and b = 0 (10')
# - [ ] 4 · Run 2,000 steps in code and plot the loss (7')
# - [ ] 5 · Compare with the exact least-squares line (5')
# - [ ] ✱ Stretch · Raise the learning rate until the loss explodes, and explain why

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

ten = U.ten_rows()
x = ten["n_medications"].to_numpy(dtype=float)
y = ten["visit_minutes"].to_numpy(dtype=float)
print(ten[["client_id", "n_medications", "visit_minutes"]].to_string(index=False))

# %% [markdown]
# ## 1 · A line by eye, scored by hand (8')
#
# A line is two numbers: a slope `w` (extra minutes per medicine) and an intercept `b` (minutes with no
# medicine). Its prediction for a client is `w * n_medications + b`.
#
# On paper: choose a `w` and a `b` by looking at the table. For each row write the error (prediction minus
# actual) and its square. The **mean squared error** is the average of the ten squares. Write your three
# numbers below (leave them as `None` if you are reading ahead).

# %%
MY_W, MY_B, MY_HAND_MSE = None, None, None   # for example 4.0, 18.0 and the loss you computed

# %% [markdown]
# ## 2 · The loss as a function (5')
#
# ✏️ **Exercise** · Write `mse(w, b)`: the mean of the squared errors over the ten rows (use the arrays `x`
# and `y`).

# %%
def mse(w: float, b: float) -> float:
    # BEGIN SOLUTION
    return float(np.mean((w * x + b - y) ** 2))
    # END SOLUTION


# %%
try:
    zero_loss = mse(0.0, 0.0)
except NotImplementedError:
    zero_loss = U.reference("d1_2_zero_loss")
U.keep("d1_2_zero_loss", zero_loss)
print(f"loss of the line w = 0, b = 0 (predict 0 minutes): {zero_loss:.1f}")
U.check(abs(zero_loss - float(np.mean(y ** 2))) < 1e-9, "with w = 0 and b = 0 the loss is the mean of the squared minutes")
if MY_W is not None:
    try:
        mine = mse(MY_W, MY_B)
        print(f"your line w = {MY_W}, b = {MY_B}: loss {mine:.1f}; your hand number {MY_HAND_MSE}")
        U.check(abs(mine - MY_HAND_MSE) < 0.5, "your hand number matches the function (within 0.5)")
    except NotImplementedError:
        pass

# %% [markdown]
# ## 3 · Two gradient steps by hand (10')
#
# The loss is a bowl over the two numbers `w` and `b`. Its slope in each direction (the **gradient**) is,
# with `e = w * x + b - y` the error of each row:
#
# - slope in `w`: `2 × mean(e × x)`
# - slope in `b`: `2 × mean(e)`
#
# (The pre-work derived these from the derivative of a square.) A **gradient step** moves each number a little
# against its slope: `w ← w − rate × slope_w` and `b ← b − rate × slope_b`.
#
# On paper, start from `w = 0`, `b = 0` with `rate = 0.01`. At that start every error is `−y`, so the slopes
# are `−2 × mean(x × y)` and `−2 × mean(y)`. Compute them, take the step, then compute the loss of the new line
# with your function. Take a second step from there in the next cell's output, by hand, and check.
#
# ✏️ **Exercise** · Write `gradient(w, b)` returning the pair `(slope_w, slope_b)`.

# %%
def gradient(w: float, b: float) -> tuple[float, float]:
    # BEGIN SOLUTION
    e = w * x + b - y
    return float(2 * np.mean(e * x)), float(2 * np.mean(e))
    # END SOLUTION


# %%
RATE = 0.01
try:
    g0 = gradient(0.0, 0.0)
except NotImplementedError:
    g0 = tuple(U.reference("d1_2_g0"))
U.keep("d1_2_g0", list(g0))
w1, b1 = 0.0 - RATE * g0[0], 0.0 - RATE * g0[1]
try:
    g1 = gradient(w1, b1)
except NotImplementedError:
    g1 = tuple(U.reference("d1_2_g1"))
U.keep("d1_2_g1", list(g1))
w2, b2 = w1 - RATE * g1[0], b1 - RATE * g1[1]
steps = pd.DataFrame({"w": [0.0, w1, w2], "b": [0.0, b1, b2],
                      "loss": [float(np.mean((w * x + b - y) ** 2)) for w, b in [(0, 0), (w1, b1), (w2, b2)]]},
                     index=pd.Index(["start", "step 1", "step 2"], name="line"))
print(f"slopes at the start: slope_w = {g0[0]:.1f}, slope_b = {g0[1]:.1f}")
print(steps.round(3).to_string())
U.check(steps.loss.is_monotonic_decreasing, "each step lowers the loss")
U.check(abs(g0[0] + 2 * float(np.mean(x * y))) < 1e-9, "the first slope in w is -2 × mean(x × y)")

# %% [markdown]
# 🔍 **Look at** the second step. The first step jumped far (the start was a very bad line, so the slopes were
# steep); the second is smaller in `w`. Near the bottom of the bowl the slopes flatten and the steps shrink by
# themselves, with the rate unchanged.
#
# ## 4 · Two thousand steps (7')
#
# ✏️ **Exercise** · Write `descend(w, b, rate, n)`: take `n` gradient steps from `(w, b)` and return the final
# `w`, the final `b` and the list of the `n + 1` losses (the start included).

# %%
def descend(w: float, b: float, rate: float, n: int) -> tuple[float, float, list[float]]:
    # BEGIN SOLUTION
    losses = [float(np.mean((w * x + b - y) ** 2))]
    for _ in range(n):
        gw, gb = gradient(w, b)
        w, b = w - rate * gw, b - rate * gb
        losses.append(float(np.mean((w * x + b - y) ** 2)))
    return w, b, losses
    # END SOLUTION


# %%
N = U.pick(2000, 2000)
try:
    w_gd, b_gd, losses = descend(0.0, 0.0, RATE, N)
except NotImplementedError:
    w_gd, b_gd, losses = U.reference("d1_2_descend")
U.keep("d1_2_descend", [w_gd, b_gd, losses])
print(f"after {N} steps: w = {w_gd:.3f} minutes per medicine, b = {b_gd:.3f} minutes, loss {losses[-1]:.3f}")
plt = U.style()
fig, ax = plt.subplots()
ax.plot(range(len(losses)), losses)
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("mean squared error (log scale)")
ax.set_title("Gradient descent on the ten rows")
U.savefig(fig, "D1_2_loss_curve.png")
U.check(losses[-1] < losses[2], "two thousand steps end lower than two")

# %% [markdown]
# ## 5 · The exact line (5')
#
# For a straight line and a squared error there is a formula for the bottom of the bowl (least squares; you
# compute it by hand in Lab 1.3). `numpy.polyfit` uses it.

# %%
w_exact, b_exact = np.polyfit(x, y, 1)
summary = pd.DataFrame({"w": [w_gd, w_exact], "b": [b_gd, b_exact],
                        "loss": [losses[-1], float(np.mean((w_exact * x + b_exact - y) ** 2))]},
                       index=["gradient descent", "exact least squares"])
print(summary.round(3).to_string())
U.check(abs(w_gd - w_exact) < 0.01 and abs(b_gd - b_exact) < 0.05, "gradient descent reached the exact line")
U.save("D1_2_steps.md", "# Lab 1.2\n\n" + steps.round(3).to_markdown() + "\n\n" + summary.round(3).to_markdown() + "\n")

# %% [markdown]
# Read the line: every extra medicine adds about `w` minutes to a visit, and a client with no medicine needs
# about `b` minutes. The loss that is left is the part of visit length that medicines do not explain.
#
# ## ✱ Stretch · A rate that is too large
#
# Run `descend(0.0, 0.0, rate, 50)` with rates 0.02, 0.025 and 0.03 and print the last loss of each. At one of
# them the loss grows without end. Why does a step that is too large overshoot the bottom of the bowl, and why
# does the feature with the largest values (`x` up to 10) decide where that happens?

# %% [markdown]
# ## Checkpoint
#
# 1. Why square the errors rather than add them? <details><summary>Answer</summary>Positive and negative errors
#    would cancel, so a bad line could score zero. Squares are all positive and punish large errors most.</details>
# 2. The slope in `w` is negative at the start. Which way does the step move `w`? <details><summary>Answer</summary>
#    Up: the step moves against the slope, and a negative slope means the loss falls as `w` grows.</details>
# 3. Gradient descent and the formula gave the same line. Why learn the slow way? <details><summary>Answer</summary>
#    The formula exists for a line and a squared error only. Logistic regression and neural networks have no
#    such formula; they are fitted by the same kind of steps.</details>
#
# **Carry forward.** `artifacts/D1_2_steps.md` and `artifacts/D1_2_loss_curve.png`. Lab 1.3 starts again from
# `data/ten_rows.csv`.
