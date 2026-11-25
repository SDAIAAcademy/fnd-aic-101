# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 1.1 · A screening rule by hand
#
# Day 1 · Session 1 (Features, labels and prediction rules) · 35 min.
#
# Hurbuth Home Care wants to know which older clients are likely to fall in the six months after a nurse's
# home assessment, so that the physiotherapy team can visit them first. Before any library, you write the
# rule yourself: on paper for ten clients, then as a Python function, then you count its two kinds of
# error on 1,000 assessments it was not written from.
#
# **AI policy: AI-off.** The rule and the counting are yours. After your first attempt an assistant may help
# you debug a Python error (AI-tutor), not write the rule.
#
# **Reference state.** `data/ten_rows.csv` and `data/assessments.csv`, supplied with the course. Nothing from
# another lab.
#
# **Tasks**
# - [ ] 1 · Read the ten rows: name the identifier, the features and the label (5')
# - [ ] 2 · Write a rule on paper that flags a client and apply it to the ten rows (8')
# - [ ] 3 · Turn the rule into a Python function (7')
# - [ ] 4 · Count its errors on 1,000 training assessments: missed fallers and needless flags (10')
# - [ ] 5 · Compare with a colleague's rule and with flagging nobody (5')
# - [ ] ✱ Stretch · Give your rule one number you can tune and find its best value on the training rows

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
import pandas as pd
import hurbuth as U

ten = U.ten_rows()
train, _, _ = U.split()
print(ten.to_string(index=False))

# %% [markdown]
# ## 1 · Identifier, features and label (5')
#
# Each row is one home assessment of one client. One column says who the row is about, one column is what
# Hurbuth wants to predict (did the client fall in the next six months, 1 or 0), and the others are what the
# nurse records at the visit.
#
# ✏️ **Exercise** · Write `roles(columns)`: return a dictionary with the key `"id"` (one column name),
# `"label"` (one column name) and `"features"` (the list of the other columns, in their order).

# %%
def roles(columns: list[str]) -> dict:
    # BEGIN SOLUTION
    features = [c for c in columns if c not in ("client_id", "fell_6m")]
    return {"id": "client_id", "label": "fell_6m", "features": features}
    # END SOLUTION


# %%
try:
    r = roles(list(ten.columns))
except NotImplementedError:
    r = U.reference("d1_1_roles")
U.keep("d1_1_roles", r)
print(r)
U.check(r["label"] == "fell_6m" and r["id"] == "client_id", "the label is fell_6m and the identifier is client_id")
U.check("visit_minutes" in r["features"] and len(r["features"]) == 4, "four features, visit_minutes among them")

# %% [markdown]
# 🔍 **Look at** `visit_minutes`. It is recorded at the visit, so it is available when the screen runs, and it
# is a feature here. Tomorrow it becomes a label of its own (how long will a visit take?). Whether a column is
# a feature or a label depends on the question, not on the column.
#
# ## 2 · A rule on paper (8')
#
# Close the laptop lid halfway. On paper, write a rule of the form *flag the client if ...* using
# `prior_falls_12m` (falls recorded in the year before) and `walk_seconds` (seconds to walk four metres;
# slower is longer). Apply it to the ten rows and write F (flag) or a dash beside each client id.
#
# Then count: how many of the fallers (`fell_6m = 1`) did you flag? How many non-fallers did you flag?
#
# ## 3 · The rule as a function (7')
#
# ✏️ **Exercise** · Write `rule(row)`: it receives one row (a pandas Series) and returns `True` to flag the
# client. Put your paper rule in it. The reference rule flags a client with a recorded fall in the past year
# or a walk of 8 seconds or more; yours may differ.

# %%
def rule(row) -> bool:
    # BEGIN SOLUTION
    return bool(row["prior_falls_12m"] >= 1 or row["walk_seconds"] >= 8.0)
    # END SOLUTION


# %%
try:
    flags_ten = [rule(row) for _, row in ten.iterrows()]
except NotImplementedError:
    flags_ten = U.reference("d1_1_flags_ten")
U.keep("d1_1_flags_ten", flags_ten)
print(pd.DataFrame({"client_id": ten.client_id, "flag": flags_ten, "fell_6m": ten.fell_6m}).to_string(index=False))
U.check(len(flags_ten) == 10 and all(isinstance(f, bool) for f in flags_ten), "one True or False per client")

# %% [markdown]
# ## 4 · Its errors on 1,000 assessments (10')
#
# Ten rows are the rows you wrote the rule from: it will look good on them. The fair test is rows you did not
# look at. Below are 1,000 assessments of the training period, drawn with the course's fixed seed.
#
# ✏️ **Exercise** · Write `count_errors(df, rule)`: apply the rule to every row and return a dictionary with
# `"caught"` (fallers flagged), `"missed"` (fallers not flagged), `"needless"` (non-fallers flagged) and
# `"passed"` (non-fallers not flagged).

# %%
sample = train.sample(1000, random_state=U.SEED)


def count_errors(df: pd.DataFrame, rule) -> dict:
    # BEGIN SOLUTION
    flag = df.apply(rule, axis=1).astype(bool)
    fell = df["fell_6m"] == 1
    return {"caught": int((flag & fell).sum()), "missed": int((~flag & fell).sum()),
            "needless": int((flag & ~fell).sum()), "passed": int((~flag & ~fell).sum())}
    # END SOLUTION


# %%
try:
    mine = count_errors(sample, rule)
except NotImplementedError:
    mine = U.reference("d1_1_mine")
U.keep("d1_1_mine", mine)
print(mine)
U.check(sum(mine.values()) == 1000, "the four counts add up to the 1,000 rows")
fallers = mine["caught"] + mine["missed"]
print(f"fallers in the sample: {fallers}; your rule caught {mine['caught']} and flagged "
      f"{mine['caught'] + mine['needless']} clients in all")

# %% [markdown]
# ## 5 · Against a colleague and against nobody (5')
#
# A colleague's rule uses the walk alone: flag a walk of 7 seconds or more. The laziest rule flags nobody.

# %%
def colleague(row) -> bool:
    return bool(row["walk_seconds"] >= 7.0)


def nobody(row) -> bool:
    return False


try:
    rows = {"your rule": mine, "colleague (walk ≥ 7 s)": count_errors(sample, colleague),
            "flag nobody": count_errors(sample, nobody)}
except NotImplementedError:
    rows = U.reference("d1_1_compare")
U.keep("d1_1_compare", rows)
compare = pd.DataFrame(rows).T
compare["flagged"] = compare.caught + compare.needless
compare["right"] = compare.caught + compare.passed
print(compare.to_string())
U.check(compare.loc["flag nobody", "missed"] == fallers, "flagging nobody misses every faller")
U.save("D1_1_rules.md", "# Lab 1.1 · three rules on 1,000 assessments\n\n" + compare.to_markdown() + "\n")

# %% [markdown]
# 🔍 **Look at** the column `right`. Flagging nobody gets most rows right, because most clients do not fall.
# A count of right answers says little about a screen whose job is to find the few who will. Keep the two
# kinds of error apart: a missed faller and a needless visit do not cost the same, and on Day 2 you will
# put a price on each.
#
# ## ✱ Stretch · One number to tune
#
# Rewrite your rule with one number in it (for example the walk threshold), try ten values on the 1,000 rows
# and pick the one with the fewest missed fallers for at most 300 flags. Then ask: which rows did you tune on,
# and what will that do to the number you report?

# %% [markdown]
# ## Checkpoint
#
# 1. Your rule caught most fallers in the ten rows and fewer in the 1,000. Why? <details><summary>Answer</summary>
#    You wrote it while looking at the ten rows. A rule always looks best on the rows it was written from;
#    the 1,000 rows are a fairer test.</details>
# 2. Flagging nobody is right on most rows. Why is it still a bad screen? <details><summary>Answer</summary>It
#    misses every faller, and finding fallers is the screen's job. Most rows are non-fallers, so counting right
#    answers rewards doing nothing.</details>
# 3. Is `client_id` a useful feature? <details><summary>Answer</summary>No. It names the row; it says nothing
#    about the client's risk, and a rule that used it would not carry over to new clients.</details>
#
# **Carry forward.** `artifacts/D1_1_rules.md`. Lab 1.2 starts again from `data/ten_rows.csv`.
