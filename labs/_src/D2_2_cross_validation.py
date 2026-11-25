# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.2 · Grouped folds and six curves
#
# Day 2 · Session 2 (Cross-validation and learning curves) · 35 min.
#
# One validation set gives one number. Cross-validation gives five, and their spread tells you how much one
# number can be trusted. A learning curve tells you whether more assessments would help. Then you read six
# curves, as you will in the supervised check, and name a diagnosis and a remedy for each.
#
# **AI policy: AI-off** for your first attempt and for the six diagnoses; AI-tutor for debugging after.
#
# **Reference state.** `data/assessments.csv` and the fixed split. Nothing from Lab 2.1.
#
# **Tasks**
# - [ ] 1 · Five folds grouped by client on the training period: score a depth-4 tree (8')
# - [ ] 2 · Read the spread across folds against one validation number (5')
# - [ ] 3 · Draw a learning curve from 100 rows to all training rows (8')
# - [ ] 4 · Diagnose six curves and name a remedy for each (14')
# - [ ] ✱ Stretch · Compare grouped folds with plain folds on the same rows

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
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

train, val, _ = U.split()
train = train.sample(frac=1.0, random_state=U.SEED)       # a fixed shuffle, so subsets are not one season
X_tr, y_tr = train[U.FEATURES], train.fell_6m
print(f"training rows {len(train)} from {train.client_id.nunique()} clients; "
      f"{(train.client_id.value_counts() > 1).sum()} clients have more than one assessment")

# %% [markdown]
# ## 1 · Five folds, grouped by client (8')
#
# Cross-validation cuts the training rows into five folds, fits on four and scores the fifth, five times. Some
# clients were assessed twice in the training period. If one assessment sits in the fitting folds and the other
# in the scored fold, the model has partly seen the client. **Grouped** folds keep every client's rows together.
#
# ✏️ **Exercise** · Write `grouped_cv(model)`: the five AUC scores of `model` under
# `GroupKFold(n_splits=5)` with `groups=train.client_id` (use `cross_val_score` with `scoring="roc_auc"`).

# %%
def grouped_cv(model) -> np.ndarray:
    # BEGIN SOLUTION
    return cross_val_score(model, X_tr, y_tr, groups=train.client_id, cv=GroupKFold(n_splits=5), scoring="roc_auc")
    # END SOLUTION


# %%
tree4 = DecisionTreeClassifier(max_depth=4, random_state=U.SEED)
try:
    folds = grouped_cv(tree4)
except NotImplementedError:
    folds = np.array(U.reference("d2_2_folds"))
U.keep("d2_2_folds", [float(f) for f in folds])
print("fold AUCs:", np.round(folds, 3), f"mean {folds.mean():.3f}, standard deviation {folds.std():.3f}")
U.check(len(folds) == 5, "five scores, one per fold")

# %% [markdown]
# ## 2 · One number against five (5')

# %%
one = float(roc_auc_score(val.fell_6m, tree4.fit(X_tr, y_tr).predict_proba(val[U.FEATURES])[:, 1]))
print(f"validation AUC of the same tree: {one:.3f}; fold range {folds.min():.3f} to {folds.max():.3f}")

# %% [markdown]
# 🔍 **Look at** the range of the five folds. If two models differ by less than that range, one validation
# number cannot tell them apart. Cross-validation does not replace the later validation and test periods here:
# it reuses the training period to say how stable a score is.
#
# ## 3 · A learning curve (8')
#
# ✏️ **Exercise** · Write `learning_curve(make_model, sizes, features)`: for each size `n`, fit a fresh
# `make_model()` on the first `n` training rows (`train.iloc[:n]`) and record its training AUC (on those `n`
# rows) and its validation AUC. Return a DataFrame indexed by size (name `rows`) with the columns `train` and
# `validation`.

# %%
SIZES = [100, 200, 400, 800, 1200, 1600, len(train)]


def learning_curve(make_model, sizes=SIZES, features=U.FEATURES) -> pd.DataFrame:
    # BEGIN SOLUTION
    out = []
    for n in sizes:
        part = train.iloc[:n]
        m = make_model().fit(part[features], part.fell_6m)
        out.append((float(roc_auc_score(part.fell_6m, m.predict_proba(part[features])[:, 1])),
                    float(roc_auc_score(val.fell_6m, m.predict_proba(val[features])[:, 1]))))
    return pd.DataFrame(out, index=pd.Index(list(sizes), name="rows"), columns=["train", "validation"])
    # END SOLUTION


# %%
def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression())


try:
    lc_logistic = learning_curve(logistic)
except NotImplementedError:
    lc_logistic = U.reference("d2_2_lc_logistic")
    lc_logistic.index = pd.Index([int(i) for i in lc_logistic.index], name="rows")
U.keep("d2_2_lc_logistic", lc_logistic)
print(lc_logistic.round(3).T.to_string())
gap_start = lc_logistic.train.iloc[0] - lc_logistic.validation.iloc[0]
gap_end = lc_logistic.train.iloc[-1] - lc_logistic.validation.iloc[-1]
U.check(gap_end < gap_start, "the gap between training and validation narrows with more rows")

# %% [markdown]
# ## 4 · Six curves (14')
#
# The cell below draws six curves of the fall screen, A to F. Four are learning curves (more rows to the
# right), two are complexity curves (more freedom to the right). Each has a **marked point**. For each, decide
# whether the model at the marked point **overfits**, **underfits** or is a **good fit**, and name one remedy.

# %%
def tree(depth=None):
    return lambda: DecisionTreeClassifier(max_depth=depth, random_state=U.SEED)


def complexity(make, values):
    out = []
    for v in values:
        m = make(v).fit(X_tr, y_tr)
        out.append((float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
                    float(roc_auc_score(val.fell_6m, m.predict_proba(val[U.FEATURES])[:, 1]))))
    return pd.DataFrame(out, index=list(values), columns=["train", "validation"])


try:
    six = {
        "A": ("learning curve, unlimited tree", learning_curve(tree(None)), len(train)),
        "B": ("learning curve, one-question tree", learning_curve(tree(1)), len(train)),
        "C": ("learning curve, logistic regression", lc_logistic, len(train)),
        "D": ("learning curve, logistic regression on two features", learning_curve(logistic, features=["stairs_at_home", "diabetes"]), len(train)),
        "E": ("complexity curve, tree depth", complexity(lambda d: DecisionTreeClassifier(max_depth=d, random_state=U.SEED), range(1, 21)), 18),
        "F": ("complexity curve, neighbours k (small k = more freedom)", complexity(lambda k: make_pipeline(StandardScaler(), KNeighborsClassifier(k)), [400, 200, 100, 50, 25, 10, 5, 3, 1]), 1),
    }
except NotImplementedError:
    raw = U.reference("d2_2_six")
    six = {k: (v[0], v[1], v[2]) for k, v in raw.items()}
U.keep("d2_2_six", {k: [v[0], v[1], v[2]] for k, v in six.items()})

plt = U.style()
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for ax, (key, (title, c, mark)) in zip(axes.flat, six.items()):
    xs = list(range(len(c))) if key == "F" else list(c.index)
    ax.plot(xs, c.train, marker="o", label="training")
    ax.plot(xs, c.validation, marker="o", label="validation")
    pos = xs[list(c.index).index(mark)] if mark in list(c.index) else xs[-1]
    ax.axvline(pos, color="grey", linestyle=":")
    if key == "F":
        ax.set_xticks(xs, [str(i) for i in c.index])
    ax.set_title(f"{key} · {title}", fontsize=9)
axes.flat[0].legend(fontsize=8)
fig.supylabel("AUC")
fig.tight_layout()
U.savefig(fig, "D2_2_six_curves.png")
for key, (title, c, mark) in six.items():
    row = c.loc[mark] if mark in c.index else c.iloc[-1]
    print(f"{key}: {title}; at the marked point training {row.train:.3f}, validation {row.validation:.3f}")

# %% [markdown]
# ✏️ **Exercise** · Write `diagnose()`: return two dictionaries keyed by letter, the diagnosis (one of
# `"overfitting"`, `"underfitting"` or `"good fit"`) and one short remedy. Do it from the plot, without an
# assistant.

# %%
def diagnose() -> tuple[dict, dict]:
    # BEGIN SOLUTION
    diagnosis = {"A": "overfitting", "B": "underfitting", "C": "good fit", "D": "underfitting",
                 "E": "overfitting", "F": "overfitting"}
    remedy = {"A": "limit depth or leaf size; more rows barely help while the gap stays this wide",
              "B": "give the model more freedom: more questions, or another family",
              "C": "keep it; more rows will not help, better features might",
              "D": "add informative features; two weak ones cannot rank fallers",
              "E": "choose a shallow depth at the validation peak",
              "F": "use a larger k, around the validation peak"}
    return diagnosis, remedy
    # END SOLUTION


try:
    DIAGNOSIS, REMEDY = diagnose()
except NotImplementedError:
    DIAGNOSIS = {k: None for k in "ABCDEF"}
    REMEDY = {k: None for k in "ABCDEF"}
    print("ℹ️  no diagnoses yet: write diagnose() above")

# %%
key = U.answer("d2_2_key") or DIAGNOSIS   # the reference key, written by the trainer's run
done = [k for k, v in DIAGNOSIS.items() if v]
right = sum(DIAGNOSIS[k] == key[k] for k in done)
print(f"{len(done)} of 6 diagnosed; {right} match the reference")
if all(DIAGNOSIS.values()):
    U.keep("d2_2_key", DIAGNOSIS)
if done:
    U.check(right >= 5, "at least five of six diagnoses match the reference (the supervised check asks the same)")
U.save("D2_2_cross_validation.md", "# Lab 2.2\n\n" + f"Grouped five-fold AUC: {np.round(folds, 3).tolist()}\n\n"
       + lc_logistic.round(3).to_markdown() + "\n\n"
       + pd.DataFrame({"diagnosis": DIAGNOSIS, "remedy": REMEDY}).to_markdown() + "\n")

# %% [markdown]
# ## ✱ Stretch · Grouped against plain folds
#
# Repeat task 1 with `KFold(n_splits=5, shuffle=True, random_state=U.SEED)` and no groups, for an unlimited
# tree. Is the mean higher or lower than with grouped folds? Which clients make the difference?

# %% [markdown]
# ## Checkpoint
#
# 1. Two models differ by 0.01 in validation AUC; the fold scores of each range over 0.05. What can you say?
#    <details><summary>Answer</summary>Not much: the difference is smaller than the noise between folds.</details>
# 2. The learning curve of your model has closed its gap at 400 rows and is flat after. Should Hurbuth collect
#    more assessments to improve it? <details><summary>Answer</summary>Not for this model: it has learnt what these
#    features can teach it. Better features or a more flexible model are the levers.</details>
# 3. Why group folds by client? <details><summary>Answer</summary>So no client has rows on both sides of a fold;
#    otherwise the score partly measures recognising a client already seen.</details>
#
# **Carry forward.** `artifacts/D2_2_six_curves.png` and `artifacts/D2_2_cross_validation.md`. Lab 2.3 starts
# from supplied scores.
