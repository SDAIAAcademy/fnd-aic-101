# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.2 · A later batch from a new sector
#
# Day 4 · Session 2 (Distribution shift) · 35 min.
#
# The screen is in use. Assessments from July to December 2025 have come in, their six-month outcomes are now
# known, and they include the **Central** sector, which Hurbuth opened in July 2025 with a newly hired team. The
# physiotherapy team lead reports that almost every Central client is being flagged. You find out why.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after.
#
# **Reference state.** `data/assessments.csv` (the screen is refitted from it in the first cell) and
# `data/later_batch.csv`. Nothing from Lab 4.1.
#
# **Tasks**
# - [ ] 1 · Score the later batch with the chosen screen (5')
# - [ ] 2 · Flag rates and recall by sector (8')
# - [ ] 3 · Compare feature distributions: training rows against the batch (10')
# - [ ] 4 · Find the cause and correct it (7')
# - [ ] 5 · Note what the model card must say (5')
# - [ ] ✱ Stretch · Compare the base rate of the new sector with the training period

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
from sklearn.metrics import roc_auc_score

train, val, _ = U.split()
screen = U.standard_screen(train)
THRESHOLD = U.cheapest_threshold(val.fell_6m, screen.predict_proba(val[U.FEATURES])[:, 1])
batch = U.later_batch()
print(f"the standard screen, threshold {THRESHOLD:.2f} chosen on validation by the costs of Lab 2.4")
print(f"later batch: {len(batch)} assessments, {batch.assessment_date.min():%Y-%m-%d} to {batch.assessment_date.max():%Y-%m-%d}")
print(batch.sector.value_counts().to_string())

# %% [markdown]
# ## 1 · Score the batch (5')

# %%
batch = batch.assign(score=screen.predict_proba(batch[U.FEATURES])[:, 1])
batch["flag"] = batch.score >= THRESHOLD
print(f"flagged {batch.flag.mean():.1%} of the batch; on validation the screen flagged "
      f"{(screen.predict_proba(val[U.FEATURES])[:, 1] >= THRESHOLD).mean():.1%}")

# %% [markdown]
# ## 2 · By sector (8')
#
# ✏️ **Exercise** · Write `by_sector(df)`: one row per sector with `assessments`, `flagged` (share),
# `recall`, `precision` and `base_rate`.

# %%
def by_sector(df: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = {}
    for sector, g in df.groupby("sector"):
        c = U.confusion_counts(g.fell_6m, g.flag)
        rows[sector] = {"assessments": len(g), "flagged": g.flag.mean(), "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
                        "precision": c["tp"] / max(c["tp"] + c["fp"], 1), "base_rate": g.fell_6m.mean()}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    sectors = by_sector(batch)
except NotImplementedError:
    sectors = U.reference("d4_2_sectors")
U.keep("d4_2_sectors", sectors)
print(sectors.round(3).to_string())
U.check(sectors.loc["Central", "flagged"] > 2 * sectors.drop(index="Central").flagged.max(),
        "Central is flagged far more often than any other sector")

# %% [markdown]
# 🔍 **Look at** Central's precision next to its base rate. Flagging nearly everyone catches nearly every faller,
# and most programmes go to clients who will not fall. Something about Central's inputs pushes every score up.
#
# ## 3 · What moved (10')
#
# ✏️ **Exercise** · Write `shift_table(reference, batch, features)`: for each feature, the mean in the reference
# rows, the mean in the batch's Central rows and the mean in the batch's other rows, and a column `ratio` (Central
# mean divided by the reference mean).

# %%
def shift_table(reference: pd.DataFrame, batch: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    # BEGIN SOLUTION
    central = batch[batch.sector == "Central"]
    others = batch[batch.sector != "Central"]
    t = pd.DataFrame({"training": reference[features].mean(), "batch, other sectors": others[features].mean(),
                      "batch, Central": central[features].mean()})
    t["ratio"] = t["batch, Central"] / t["training"]
    return t
    # END SOLUTION


# %%
try:
    shift = shift_table(train, batch, U.FEATURES)
except NotImplementedError:
    shift = U.reference("d4_2_shift")
U.keep("d4_2_shift", shift)
print(shift.round(2).to_string())

plt = U.style()
fig, ax = plt.subplots()
ax.hist(train.walk_seconds, bins=40, alpha=0.6, density=True, label="training rows")
ax.hist(batch[batch.sector == "Central"].walk_seconds, bins=40, alpha=0.6, density=True, label="batch, Central")
ax.set_xlabel("walk_seconds"); ax.set_ylabel("share of assessments"); ax.legend()
ax.set_title("Lab 4.2 · the walk test, training against Central")
U.savefig(fig, "D4_2_walk_shift.png")

# %% [markdown]
# Central's clients are older on average, which explains a little. One feature moved far more than age can
# explain. The physiotherapy team lead confirms it after you show her the plot: the new Central team was trained
# to time the walk over **six metres**, not four. Nothing about the clients is that different; the measurement is.
#
# ## 4 · Correct and rescore (7')
#
# ✏️ **Exercise** · Write `correct_walk(df)`: a copy of `df` in which Central's `walk_seconds` is converted to a
# four-metre walk (multiply by 4 / 6).

# %%
def correct_walk(df: pd.DataFrame) -> pd.DataFrame:
    # BEGIN SOLUTION
    out = df.copy()
    central = out.sector == "Central"
    out.loc[central, "walk_seconds"] = (out.loc[central, "walk_seconds"] * 4 / 6).round(1)
    return out
    # END SOLUTION


# %%
try:
    fixed = correct_walk(batch)
except NotImplementedError:
    fixed = None
if fixed is not None:
    fixed = fixed.assign(score=screen.predict_proba(fixed[U.FEATURES])[:, 1])
    fixed["flag"] = fixed.score >= THRESHOLD
    try:
        after = by_sector(fixed)
    except NotImplementedError:
        after = U.reference("d4_2_after")
else:
    after = U.reference("d4_2_after")
U.keep("d4_2_after", after)
print(after.round(3).to_string())
central_rows = batch.sector == "Central"
auc_before = roc_auc_score(batch.fell_6m[central_rows], batch.score[central_rows])
print(f"whole batch AUC before the correction: {roc_auc_score(batch.fell_6m, batch.score):.3f}"
      + (f"; after: {roc_auc_score(fixed.fell_6m, fixed.score):.3f}" if fixed is not None else ""))
print(f"Central's own AUC before: {auc_before:.3f}"
      + (f"; after: {roc_auc_score(fixed.fell_6m[central_rows], fixed.score[central_rows]):.3f}" if fixed is not None else ""))
U.check(after.loc["Central", "flagged"] < 0.6 * sectors.loc["Central", "flagged"], "the correction brings Central's flag rate down")
U.save("D4_2_later_batch.md", f"# Lab 4.2 · later batch, threshold {THRESHOLD:.2f}\n\nBefore the correction:\n\n"
       + sectors.round(3).to_markdown() + "\n\nWhat moved:\n\n" + shift.round(2).to_markdown()
       + "\n\nAfter converting Central's walk to four metres:\n\n" + after.round(3).to_markdown() + "\n")

# %% [markdown]
# Central's AUC inside its own rows does not change with the correction (multiplying one column by a constant keeps
# the order of Central clients); the flags change completely. Mixed with the other sectors, the unconverted rows
# also pull the whole batch's AUC down, because Central clients are ranked above everyone else. A shift in one input's unit is invisible to a ranking score and obvious in
# the flag rate. Monitor both.
#
# ## 5 · For the model card (5')
#
# Write two or three lines for the "known limits" of the model card: what the screen assumes about how the walk is
# measured, which check would have caught this in the first week (a flag rate or a feature mean by sector against
# the training period), and who acts on the alert (the physiotherapy team lead, with the analyst; the clinical
# director decides whether the screen is paused for a sector).

# %% [markdown]
# *Your lines here.*

# %% [markdown]
# ## ✱ Stretch · Base rates
#
# Compare Central's base rate with the training period's. If it stays higher after the correction, what would you
# ask about Central's clients, and would you change the threshold for one sector only?

# %% [markdown]
# ## Checkpoint
#
# 1. Central's own AUC is the same before and after the correction. Why was the problem still serious?
#    <details><summary>Answer</summary>
#    AUC measures ranking; the threshold turns scores into programmes. A unit change moved every Central score
#    above the threshold.</details>
# 2. Which monitoring check is cheapest and catches this? <details><summary>Answer</summary>The share flagged and
#    the mean of each input, per sector and per month, against the training period.</details>
# 3. Should the analyst simply rescale the column in production? <details><summary>Answer</summary>Only after the
#    measuring protocol is confirmed by the team that owns it, and with the change recorded and reviewed by
#    someone other than the analyst.</details>
#
# **Carry forward.** `artifacts/D4_2_later_batch.md`. Lab 4.3 refits the standard screen itself.
