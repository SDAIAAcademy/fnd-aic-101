# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.4 · A threshold from costs
#
# Day 2 · Session 4 (Thresholds, error costs and base rates) · 40 min. **Graded lab.**
#
# The physiotherapy team can run a home programme for any client the screen flags. A needless programme costs
# **1 unit**; a fall that the screen missed costs, on average, **6 units** (the case's assumption, agreed with the
# clinical director). You choose the threshold that makes the expected cost lowest on the validation scores, then
# report it once on the test scores, and you write who approves it.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/scores/val_scores.csv` and `data/scores/test_scores.csv`, supplied. Nothing from
# Lab 2.3.
#
# **What is graded.** The cost sweep, the chosen threshold, the metrics table on the test scores, and the three
# sentences of task 6 with an owner and a different reviewer.
#
# **Tasks**
# - [ ] 1 · Expected cost of a threshold (8')
# - [ ] 2 · Sweep validation thresholds for the cheapest (8')
# - [ ] 3 · Compare with the cost-ratio rule (6')
# - [ ] 4 · Apply it once to the test scores (8')
# - [ ] 5 · Raise the base rate: what moves (5')
# - [ ] 6 · Justify it, and name who approves (5')
# - [ ] ✱ Stretch · Add a capacity limit of one client in four

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

val, test = U.scores("val"), U.scores("test")
C_FN, C_FP = U.COST_MISSED_FALL, U.COST_NEEDLESS_PROGRAMME
print(f"validation {len(val)} scores, base rate {val.fell_6m.mean():.3f}; test {len(test)} scores")
print(f"cost of a missed fall {C_FN} units, of a needless programme {C_FP} unit")

# %% [markdown]
# ## 1 · The cost of one threshold (8')
#
# The expected cost per client is `(C_FN × FN + C_FP × FP) / clients`. Multiply by 1,000 to read it per 1,000
# clients. Correct passes and caught fallers cost nothing here (the programme for a faller is money well spent).
#
# ✏️ **Exercise** · Write `cost_per_1000(y, score, t)`. Use `U.confusion_counts(y, score >= t)` for the cells.

# %%
def cost_per_1000(y, score, t: float) -> float:
    # BEGIN SOLUTION
    c = U.confusion_counts(y, np.asarray(score) >= t)
    return 1000 * (C_FN * c["fn"] + C_FP * c["fp"]) / len(np.asarray(y))
    # END SOLUTION


# %%
try:
    at_half = cost_per_1000(val.fell_6m, val.score, 0.5)
    nobody, everyone = cost_per_1000(val.fell_6m, val.score, 1.01), cost_per_1000(val.fell_6m, val.score, 0.0)
except NotImplementedError:
    at_half, nobody, everyone = U.reference("d2_4_three_costs")
U.keep("d2_4_three_costs", [at_half, nobody, everyone])
print(f"per 1,000 clients: threshold 0.5 costs {at_half:.0f}; flag nobody {nobody:.0f}; flag everyone {everyone:.0f}")
U.check(abs(nobody - 1000 * C_FN * val.fell_6m.mean()) < 1e-6, "flagging nobody costs C_FN times the base rate")

# %% [markdown]
# ## 2 · The cheapest threshold on validation (8')
#
# ✏️ **Exercise** · Write `sweep(y, score, thresholds)`: a DataFrame indexed by threshold with the columns
# `cost_per_1000`, `flagged` (share flagged), `recall` and `precision`.

# %%
def sweep(y, score, thresholds) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = {}
    for t in thresholds:
        c = U.confusion_counts(y, np.asarray(score) >= t)
        flagged = c["tp"] + c["fp"]
        rows[round(float(t), 3)] = {"cost_per_1000": cost_per_1000(y, score, t), "flagged": flagged / len(np.asarray(y)),
                                    "recall": c["tp"] / (c["tp"] + c["fn"]), "precision": c["tp"] / flagged if flagged else 0.0}
    return pd.DataFrame(rows).T.rename_axis("threshold")
    # END SOLUTION


# %%
grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
try:
    table = sweep(val.fell_6m, val.score, grid)
except NotImplementedError:
    table = U.reference("d2_4_sweep")
    table.index = pd.Index([float(i) for i in table.index], name="threshold")
U.keep("d2_4_sweep", table)
best_t = float(table.cost_per_1000.idxmin())
print(table.loc[[0.05, 0.1, 0.14, 0.15, 0.2, 0.3, 0.5]].round(3).to_string())
print(f"cheapest on validation: threshold {best_t:.2f}, {table.cost_per_1000.min():.0f} per 1,000")

plt = U.style()
fig, ax = plt.subplots()
ax.plot(table.index, table.cost_per_1000)
ax.axvline(best_t, linestyle=":", color="grey")
ax.set_xlabel("threshold"); ax.set_ylabel("expected cost per 1,000 clients (units)")
ax.set_title("Lab 2.4 · cost of each threshold on validation")
U.savefig(fig, "D2_4_cost_curve.png")
U.check(table.cost_per_1000.min() < min(nobody, everyone), "the best threshold beats flagging nobody and everyone")

# %% [markdown]
# ## 3 · The cost-ratio rule (6')
#
# If the scores were exact probabilities, flagging a client with probability `p` costs `C_FP × (1 − p)` in
# expectation and not flagging costs `C_FN × p`. Flag when the second is larger: when `p > C_FP / (C_FP + C_FN)`.

# %%
rule_t = C_FP / (C_FP + C_FN)
print(f"cost-ratio rule: {rule_t:.3f}; cheapest on validation: {best_t:.2f}")
try:
    print(f"cost at the rule: {cost_per_1000(val.fell_6m, val.score, rule_t):.0f} per 1,000")
except NotImplementedError:
    print("finish cost_per_1000 in task 1 to see the cost at the rule")

# %% [markdown]
# 🔍 **Look at** how close the two thresholds are. The rule needs no data beyond the costs, but it trusts the
# scores to be honest probabilities; the sweep trusts the validation rows. When they disagree a lot, the scores
# are badly calibrated, and that is a finding in itself.
#
# ## 4 · Once, on the test scores (8')
#
# ✏️ **Exercise** · Write `metrics_table(y, score, t)`: a one-row DataFrame with `threshold`, `flagged`,
# `recall`, `precision`, `specificity` and `cost_per_1000`.

# %%
def metrics_table(y, score, t: float) -> pd.DataFrame:
    # BEGIN SOLUTION
    c = U.confusion_counts(y, np.asarray(score) >= t)
    n = sum(c.values())
    return pd.DataFrame([{"threshold": t, "flagged": (c["tp"] + c["fp"]) / n, "recall": c["tp"] / (c["tp"] + c["fn"]),
                          "precision": c["tp"] / max(c["tp"] + c["fp"], 1), "specificity": c["tn"] / (c["tn"] + c["fp"]),
                          "cost_per_1000": cost_per_1000(y, score, t), **c}])
    # END SOLUTION


# %%
try:
    report = metrics_table(test.fell_6m, test.score, best_t)
except NotImplementedError:
    report = U.reference("d2_4_report")
U.keep("d2_4_report", report)
print(report.round(3).to_string(index=False))
U.save("D2_4_threshold.md", "# Lab 2.4 · threshold chosen on validation, reported on test\n\n"
       + f"Costs: missed fall {C_FN}, needless programme {C_FP}. Cost-ratio rule {rule_t:.3f}; "
       f"cheapest on validation {best_t:.2f}.\n\n" + report.round(3).to_markdown(index=False) + "\n")

# %% [markdown]
# ## 5 · A higher base rate (5')
#
# Suppose a sector where one client in three falls. The screen ranks clients as well as before, but there are
# more fallers. Below, every faller of the test scores is counted twice (a quick way to raise the base rate
# without changing the scores of anybody).

# %%
heavy = pd.concat([test, test[test.fell_6m == 1]], ignore_index=True)
try:
    heavy_sweep = sweep(heavy.fell_6m, heavy.score, grid)
    heavy_t = float(heavy_sweep.cost_per_1000.idxmin())
    print(f"base rate {heavy.fell_6m.mean():.3f}: the cheapest threshold is {heavy_t:.2f} (it was {best_t:.2f}); "
          f"precision at {best_t:.2f} is {heavy_sweep.loc[best_t, 'precision']:.3f} against {report.precision.iloc[0]:.3f}")
except NotImplementedError:
    print("finish sweep in task 2 to see what a higher base rate changes")

# %% [markdown]
# Two things moved. Precision rose, because more of the flagged clients are fallers. And the cost of a missed
# fall now weighs more often, so the cheapest threshold changes. The costs did not change; the base rate did.
#
# ## 6 · The justification (5')
#
# Write three sentences for the clinical director: the threshold and the costs it rests on; what it means in
# clients flagged and fallers missed per 1,000 on the test scores; who approves it and who reviews it (the
# analyst proposes, the clinical director approves, the physiotherapy team lead confirms the capacity: three
# different people).

# %% [markdown]
# *Your three sentences here.*

# %% [markdown]
# ## ✱ Stretch · A capacity limit
#
# The physiotherapy team can take at most one client in four. Find the lowest threshold that flags no more than
# 25% of the validation clients, and its cost against the unconstrained best. Who should decide between the two?

# %% [markdown]
# ## Checkpoint
#
# 1. Why not choose the threshold on the test scores? <details><summary>Answer</summary>Then the reported cost
#    would be the best of 59 tries on the same rows, an optimistic number.</details>
# 2. The clinical director raises the cost of a missed fall from 6 to 10. Does the threshold go up or down?
#    <details><summary>Answer</summary>Down: missing a faller is now dearer, so flagging more clients pays. The
#    rule gives 1 / 11, about 0.09.</details>
# 3. Who should own the costs, the analyst or the director? <details><summary>Answer</summary>The director: they
#    are a judgement about care and money. The analyst shows what each choice implies.</details>
#
# **Carry forward.** `artifacts/D2_4_threshold.md` and `artifacts/D2_4_cost_curve.png` (graded). Lab 4.3 uses
# a threshold chosen the same way.
