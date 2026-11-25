# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 2.1 · Overfit on purpose
#
# Day 2 · Session 1 (Model complexity, overfitting and regularisation) · 40 min. **Graded lab.**
#
# You give a decision tree more and more questions to ask and watch two scores: on the assessments it learns
# from and on validation assessments it never sees. Then you do the opposite to a logistic regression with far
# too many features: you hold it back with regularisation. You choose on validation and open the test set once.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after. The reading of the curves is yours.
#
# **Reference state.** `data/assessments.csv` and the fixed split. Nothing from Day 1's labs.
#
# **What is graded.** The plot with the three zones marked, the chosen depth with its validation and test AUC,
# the regularisation table, and the three-sentence diagnosis.
#
# **Tasks**
# - [ ] 1 · A one-question tree and an unlimited tree: training and validation scores (6')
# - [ ] 2 · Scores for depths 1 to 20 (8')
# - [ ] 3 · Plot the two curves and mark underfitting, the best depth and overfitting (8')
# - [ ] 4 · Regularise a logistic regression from strong to weak (8')
# - [ ] 5 · Choose the depth on validation and score the test set once (5')
# - [ ] 6 · Write the diagnosis in three sentences (5')
# - [ ] ✱ Stretch · Limit the leaf size in place of the depth and compare the best validation score

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
import warnings
import pandas as pd
import hurbuth as U
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
train, val, test = U.split()
X_tr, y_tr = train[U.FEATURES], train.fell_6m
X_va, y_va = val[U.FEATURES], val.fell_6m
X_te, y_te = test[U.FEATURES], test.fell_6m
print(f"training {len(train)}, validation {len(val)}, test {len(test)} assessments")

# %% [markdown]
# ## 1 · Two trees (6')
#
# A decision tree asks yes-or-no questions of the features ("walk over 7.1 seconds?"). `max_depth` caps how
# many questions it may ask in a row; `None` lets it go on until every training row sits in a pure group.
#
# ✏️ **Exercise** · Write `tree_auc(depth)`: fit `DecisionTreeClassifier(max_depth=depth, random_state=U.SEED)`
# on the training rows and return the pair (training AUC, validation AUC).

# %%
def tree_auc(depth) -> tuple[float, float]:
    # BEGIN SOLUTION
    t = DecisionTreeClassifier(max_depth=depth, random_state=U.SEED).fit(X_tr, y_tr)
    return (float(roc_auc_score(y_tr, t.predict_proba(X_tr)[:, 1])),
            float(roc_auc_score(y_va, t.predict_proba(X_va)[:, 1])))
    # END SOLUTION


# %%
try:
    stump, full = tree_auc(1), tree_auc(None)
except NotImplementedError:
    stump, full = U.reference("d2_1_two_trees")
U.keep("d2_1_two_trees", [stump, full])
print(f"one question:  training AUC {stump[0]:.3f}, validation AUC {stump[1]:.3f}")
print(f"no limit:      training AUC {full[0]:.3f}, validation AUC {full[1]:.3f}")
U.check(full[0] > 0.99, "the unlimited tree ranks its own training rows almost perfectly")
U.check(full[1] < stump[1] + 0.05, "on validation it is no better than one question")

# %% [markdown]
# 🔍 **Look at** the unlimited tree: almost perfect on its own rows, ordinary on new ones. It has learnt the
# training assessments themselves, noise included. That is **overfitting**.
#
# ## 2 · Depths 1 to 20 (8')
#
# ✏️ **Exercise** · Write `depth_curve(depths)`: a DataFrame indexed by depth (index name `depth`) with the
# columns `train` and `validation`.

# %%
def depth_curve(depths) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = [tree_auc(d) for d in depths]
    return pd.DataFrame(rows, index=pd.Index(list(depths), name="depth"), columns=["train", "validation"])
    # END SOLUTION


# %%
depths = range(1, 21)
try:
    curve = depth_curve(depths)
except NotImplementedError:
    curve = U.reference("d2_1_curve")
    curve.index = pd.Index([int(i) for i in curve.index], name="depth")
U.keep("d2_1_curve", curve)
print(curve.round(3).T.to_string())
U.check(curve.train.diff().dropna().ge(-1e-9).all(), "the training score never falls as the tree deepens")

# %% [markdown]
# ## 3 · The two curves (8')

# %%
plt = U.style()
fig, ax = plt.subplots()
ax.plot(curve.index, curve.train, marker="o", label="training")
ax.plot(curve.index, curve.validation, marker="o", label="validation")
ax.axhline(0.5, linestyle="--", color="grey", label="a coin (AUC 0.5)")
ax.set_xlabel("tree depth (questions in a row)"); ax.set_ylabel("AUC"); ax.set_xticks(list(curve.index)[::2])
ax.legend(); ax.set_title("Lab 2.1 · training and validation AUC by depth")
U.savefig(fig, "D2_1_depth_curves.png")

# %% [markdown]
# On your copy of the plot (or on paper), mark three zones: **underfitting** (both curves low), the **best
# depth** (validation at its peak), **overfitting** (training still rising, validation falling, the gap
# widening).
#
# ## 4 · Regularisation (8')
#
# The other way to control complexity is to keep a flexible model and hold it back. Here a logistic regression
# gets every product of up to three features (220 columns) and only 600 training rows: plenty of room to
# overfit. Its setting `C` is the inverse of the brake: small `C` holds the weights close to zero, large `C`
# lets them grow.
#
# ✏️ **Exercise** · Write `regularisation_curve(Cs)`: for each `C` fit
# `make_pipeline(PolynomialFeatures(3), StandardScaler(), LogisticRegression(C=C, max_iter=5000))` on `small`
# and return a DataFrame indexed by `C` with the columns `train` (AUC on `small`) and `validation`.

# %%
small = train.sample(600, random_state=U.SEED)


def regularisation_curve(Cs) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = []
    for C in Cs:
        m = make_pipeline(PolynomialFeatures(3), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        m.fit(small[U.FEATURES], small.fell_6m)
        rows.append((float(roc_auc_score(small.fell_6m, m.predict_proba(small[U.FEATURES])[:, 1])),
                     float(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))))
    return pd.DataFrame(rows, index=pd.Index(list(Cs), name="C"), columns=["train", "validation"])
    # END SOLUTION


# %%
Cs = U.pick([0.001, 0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.1, 10, 1000])
try:
    reg = regularisation_curve(Cs)
except NotImplementedError:
    reg = U.reference("d2_1_regularisation")
    reg.index = pd.Index([float(i) for i in reg.index], name="C")
U.keep("d2_1_regularisation", reg)
print(reg.round(3).to_string())
U.check(reg.validation.iloc[0] > reg.validation.iloc[-1], "the strongest brake beats the weakest on validation")
U.check(reg.train.iloc[-1] > reg.train.iloc[0], "the weakest brake scores best on its own training rows")

# %% [markdown]
# 🔍 **Look at** the two columns moving in opposite directions as `C` grows. That is the same picture as the
# depth curve, read from the other side: more freedom, better on the rows it saw, worse on the rows it did not.
#
# ## 5 · Choose on validation, test once (5')
#
# ✏️ **Exercise** · Write `choose_depth(curve)`: the depth with the highest validation AUC (the smallest such
# depth if two tie).

# %%
def choose_depth(curve: pd.DataFrame) -> int:
    # BEGIN SOLUTION
    best = curve.validation.max()
    return int(curve.index[curve.validation == best].min())
    # END SOLUTION


# %%
try:
    best_depth = choose_depth(curve)
except NotImplementedError:
    best_depth = U.reference("d2_1_best_depth")
U.keep("d2_1_best_depth", best_depth)
final = DecisionTreeClassifier(max_depth=best_depth, random_state=U.SEED).fit(X_tr, y_tr)
test_auc = float(roc_auc_score(y_te, final.predict_proba(X_te)[:, 1]))
print(f"chosen depth {best_depth}: validation AUC {curve.validation[best_depth]:.3f}, test AUC {test_auc:.3f}")
U.check(best_depth <= 8, "the chosen depth is a shallow tree")
U.save("D2_1_overfit.md", "# Lab 2.1\n\n" + curve.round(3).to_markdown() + "\n\n" + reg.round(3).to_markdown()
       + f"\n\nChosen depth {best_depth}; validation AUC {curve.validation[best_depth]:.3f}; test AUC {test_auc:.3f}.\n")

# %% [markdown]
# ## 6 · The diagnosis (5')
#
# Write three sentences in the cell below: where the tree underfits and why, where it overfits and what the
# evidence is, and what you would tell the clinical director about the depth you chose and its test score.

# %% [markdown]
# *Your three sentences here.*

# %% [markdown]
# ## ✱ Stretch · Limit the leaf size
#
# `min_samples_leaf` is another brake: every final group of the tree must hold at least that many training rows.
# Fit unlimited-depth trees with `min_samples_leaf` in 1, 5, 20, 50, 100 and 200. Which value gives the best
# validation AUC, and is it better than the best depth?

# %% [markdown]
# ## Checkpoint
#
# 1. Training AUC 1.00, validation AUC 0.58. What happened, and what would you try first? <details><summary>
#    Answer</summary>Overfitting: the model memorised its rows. Reduce its freedom (a shallower tree, a larger
#    leaf size, stronger regularisation) or give it more rows.</details>
# 2. Both curves are low and close together. Is more regularisation the remedy? <details><summary>Answer
#    </summary>No. That is underfitting: the model is too simple. Give it more freedom or better features.
#    </details>
# 3. Why is the validation score of the chosen depth slightly flattering? <details><summary>Answer</summary>You
#    picked it because it was the best of twenty; the test score had no such advantage.</details>
#
# **Carry forward.** `artifacts/D2_1_depth_curves.png` and `artifacts/D2_1_overfit.md` (graded). Lab 2.2 starts
# from the same fixed split.
