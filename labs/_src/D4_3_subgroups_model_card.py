# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6", "matplotlib>=3.9", "tabulate>=0.9"]
# ///
# %% [markdown]
# # Lab 4.3 · Subgroup error rates and a model card
#
# Day 4 · Session 3 (Subgroup error gaps and the model card) · 40 min. **Graded lab.**
#
# The standard screen does not use `sex` as an input. That does not make its errors the same for women and men.
# You measure the error rates by sex at the chosen threshold, with their counts, show how another column carries
# sex, and draft the screen's model card.
#
# **AI policy: AI-off** for your first attempt; AI-tutor for debugging after. The model card is yours.
#
# **Reference state.** `data/assessments.csv`, the fixed split, and `model_card_template.md`. Nothing from Lab 4.2
# (its finding is restated where the model card needs it).
#
# **What is graded.** `artifacts/D4_3_model_card.md`: the rates by sex with counts, the proxy evidence, and the
# section on who should not rely on the screen.
#
# **Tasks**
# - [ ] 1 · Error rates by sex at the chosen threshold (10')
# - [ ] 2 · The counts behind each rate (6')
# - [ ] 3 · The proxy: predict sex from nurse preference (8')
# - [ ] 4 · Draft the model card from the template (12')
# - [ ] 5 · Write who should not rely on the screen (4')
# - [ ] ✱ Stretch · Add nurse preference to the screen and compare the gap

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

train, val, test = U.split()
screen = U.standard_screen(train)
THRESHOLD = U.cheapest_threshold(val.fell_6m, screen.predict_proba(val[U.FEATURES])[:, 1])
test = test.assign(score=screen.predict_proba(test[U.FEATURES])[:, 1])
test["flag"] = test.score >= THRESHOLD
print(f"standard screen; threshold {THRESHOLD:.2f} chosen on validation; test period {len(test)} assessments")
print(f"inputs: {U.FEATURES}  (sex is not one of them)")

# %% [markdown]
# ## 1 · Error rates by sex (10')
#
# Two rates matter for a screen, and they fail people in different ways:
#
# - **recall** (of the people who fell, the share flagged): a low recall means fallers are missed;
# - **false positive rate** (of the people who did not fall, the share flagged): a high one means needless
#   programmes and visits.
#
# ✏️ **Exercise** · Write `rates_by_group(df, group)`: one row per value of `group`, with `assessments`,
# `base_rate`, `recall`, `false_positive_rate` and `flagged`, from the columns `fell_6m` and `flag`.

# %%
def rates_by_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    # BEGIN SOLUTION
    rows = {}
    for value, g in df.groupby(group):
        c = U.confusion_counts(g.fell_6m, g.flag)
        rows[value] = {"assessments": len(g), "base_rate": g.fell_6m.mean(),
                       "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
                       "false_positive_rate": c["fp"] / max(c["fp"] + c["tn"], 1), "flagged": g.flag.mean()}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    rates = rates_by_group(test, "sex")
except NotImplementedError:
    rates = U.reference("d4_3_rates")
U.keep("d4_3_rates", rates)
print(rates.round(3).to_string())
print(f"AUC within each group: " + ", ".join(
    f"{s} {roc_auc_score(g.fell_6m, g.score):.3f}" for s, g in test.groupby("sex")))
U.check(rates.loc["F", "recall"] - rates.loc["M", "recall"] > 0.1, "the recall gap between women and men is large")

# %% [markdown]
# 🔍 **Look at** both columns at once. One group has the lower recall *and* the lower false positive rate: the
# screen is more cautious with that group, so it misses more of its fallers.
#
# ## 2 · The counts behind each rate (6')
#
# A rate without its count can mislead. Write out the four cells for each group.

# %%
counts = pd.DataFrame({s: U.confusion_counts(g.fell_6m, g.flag) for s, g in test.groupby("sex")}).T
counts.columns = ["caught fallers (tp)", "needless flags (fp)", "missed fallers (fn)", "correct passes (tn)"]
print(counts.to_string())

# %% [markdown]
# ✏️ **Exercise** · How many more fallers would have been caught among men if their recall equalled women's? Use
# the counts above, then compare with the number of fallers missed in that group.

# %%
def extra_caught(counts: pd.DataFrame, rates: pd.DataFrame) -> float:
    # BEGIN SOLUTION
    fallers_m = counts.loc["M", "caught fallers (tp)"] + counts.loc["M", "missed fallers (fn)"]
    return float(fallers_m * rates.loc["F", "recall"] - counts.loc["M", "caught fallers (tp)"])
    # END SOLUTION


# %%
try:
    extra = extra_caught(counts, rates)
except NotImplementedError:
    extra = U.reference("d4_3_extra")
U.keep("d4_3_extra", extra)
print(f"about {extra:.0f} more male fallers would have been caught at women's recall "
      f"(of {counts.loc['M', 'missed fallers (fn)']} missed)")

# %% [markdown]
# Why? The screen leans on `prior_falls_12m`. Compare the recorded prior falls of clients who went on to fall:

# %%
recorded = test.groupby(["sex", "fell_6m"]).prior_falls_12m.mean().unstack()
recorded.columns = ["did not fall", "fell"]
print("mean prior_falls_12m recorded at the visit, test period")
print(recorded.round(2).to_string())

# %% [markdown]
# Men who went on to fall had far fewer prior falls *recorded*. In this synthetic case the generator planted that
# pattern: it is not a finding about real men or women. In real data the cause could be how the question is asked,
# who answers it or how notes are kept, and the data cannot show which: there is no column of true past falls. It
# is a question for the clinical team, and it belongs in the model card's known limits.
#
# ## 3 · The proxy (8')
#
# Leaving `sex` out of the inputs does not remove it if another column carries it. The SDAIA AI Ethics Principles
# ask that sensitive attributes are not model inputs unless there is a justification. Looking for columns that
# stand in for them is this course's own rule (the English edition of the Principles names it). `nurse_preference` is recorded at registration.
#
# ✏️ **Exercise** · Write `proxy_accuracy(tr, te)`: fit a logistic regression that predicts `sex == "M"` from the
# one-hot encoding of `nurse_preference` on `tr`, and return its accuracy on `te`, next to the accuracy of always
# guessing the larger group of `tr`.

# %%
def proxy_accuracy(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[float, float]:
    # BEGIN SOLUTION
    X_tr = pd.get_dummies(tr.nurse_preference, dtype=float)
    X_te = pd.get_dummies(te.nurse_preference, dtype=float).reindex(columns=X_tr.columns, fill_value=0.0)
    model = LogisticRegression().fit(X_tr, tr.sex == "M")
    accuracy = float((model.predict(X_te) == (te.sex == "M")).mean())
    majority = tr.sex.value_counts().idxmax()
    return accuracy, float((te.sex == majority).mean())
    # END SOLUTION


# %%
try:
    proxy, guess = proxy_accuracy(train, test)
except NotImplementedError:
    proxy, guess = U.reference("d4_3_proxy")
U.keep("d4_3_proxy", [proxy, guess])
print(pd.crosstab(test.sex, test.nurse_preference).to_string())
print(f"sex predicted from nurse_preference: accuracy {proxy:.3f}; always guessing the larger group: {guess:.3f}")
U.check(proxy > guess + 0.2, "nurse preference carries most of the information in sex")

# %% [markdown]
# ## 4 · Draft the model card (12')
#
# Open `model_card_template.md`. The cell below fills the fields that come straight from your numbers. Edit the
# text fields in `CARD` (intended use, out of scope, known limits, monitoring) in your own words, then run it.

# %%
CARD = {
    "intended use": f"Rank clients after a home assessment so that the physiotherapy team offers a fall-prevention "
                    f"programme to those flagged at threshold {THRESHOLD:.2f} (a missed fall costs about six times a "
                    f"needless programme). A nurse or physiotherapist reviews every flag; the score never refuses care.",
    "out of scope": "Deciding eligibility for services, staffing or any decision about an individual nurse; clients "
                    "under 65; use without the four-metre walk protocol.",
    "known limits": "Recorded prior falls are lower for men who went on to fall, and the recall for men is lower. "
                    "The walk test is assumed to be timed over four metres: a sector that timed six metres was flagged "
                    "almost entirely (Lab 4.2). Test period of one half-year.",
    "monitoring": "Monthly, by sector and by sex: share flagged, recall once outcomes arrive, and the mean of each "
                  "input against the training period. The analyst prepares it; the quality and data protection "
                  "officer reviews it; the clinical director decides on a pause.",
}
val_auc = roc_auc_score(val.fell_6m, screen.predict_proba(val[U.FEATURES])[:, 1])
test_auc = roc_auc_score(test.fell_6m, test.score)
try:
    overall = rates_by_group(test.assign(all="all"), "all")
except NotImplementedError:
    overall = None


def model_card(card: dict, who_not: str) -> str:
    lines = [
        "# Model card: Hurbuth fall screen (standard screen)", "",
        f"**Model.** Logistic regression on scaled inputs {', '.join(U.FEATURES)}; fitted on the training period "
        f"with `hurbuth.standard_screen`, seed {U.SEED}; threshold {THRESHOLD:.2f}.", "",
        "**Owner and reviewer.** Prepared by the analyst; reviewed by the quality and data protection officer; "
        "accepted for use by the clinical director.", "",
        f"**Intended use.** {card['intended use']}", "",
        f"**Out of scope.** {card['out of scope']}", "",
        f"**Data.** Synthetic home assessments of Hurbuth Home Care, January 2023 to June 2025; training before "
        f"July 2024 ({len(train)}), validation July to December 2024 ({len(val)}), test 2025 ({len(test)}).", "",
        f"**Metrics.** AUC on validation {val_auc:.3f}, on test {test_auc:.3f}; a baseline ranks at 0.500."
        + (f" At the threshold on test: recall {overall.loc['all', 'recall']:.3f}, false positive rate "
           f"{overall.loc['all', 'false_positive_rate']:.3f}, flagged {overall.loc['all', 'flagged']:.3f}."
           if overall is not None else ""), "",
        "**Metrics by subgroup (test period).**", "", rates.round(3).to_markdown(), "", counts.to_markdown(), "",
        f"**Sensitive attributes and proxies.** `sex` is not an input. `nurse_preference` predicts sex with accuracy "
        f"{proxy:.3f} (guessing the larger group: {guess:.3f}), so it is not an input either.", "",
        f"**Known limits.** {card['known limits']}", "",
        f"**Who should not rely on it.** {who_not}", "",
        f"**Monitoring.** {card['monitoring']}", "",
    ]
    return "\n".join(lines)


# %% [markdown]
# ## 5 · Who should not rely on the screen (4')
#
# ✏️ **Exercise** · Make `who_should_not_rely()` return two or three sentences: for whom is a low score weak
# evidence, and what should staff do instead? Use your numbers.

# %%
def who_should_not_rely() -> str:
    # BEGIN SOLUTION
    return (f"Male clients: the screen catches about {rates.loc['M', 'recall']:.0%} of men who will fall against "
            f"{rates.loc['F', 'recall']:.0%} of women, so a low score for a man is weaker evidence; nurses should ask "
            "about past falls directly and use their judgement. Clients assessed in a sector or with a walk protocol "
            "the screen was not trained on, until monitoring shows its rates hold there.")
    # END SOLUTION


# %%
try:
    who_not = who_should_not_rely()
except NotImplementedError:
    who_not = "(not written yet)"
U.save("D4_3_model_card.md", model_card(CARD, who_not))
U.check(who_not != "(not written yet)", "the section on who should not rely on the screen is written")

# %% [markdown]
# ## ✱ Stretch · Add the proxy
#
# Refit the screen with `nurse_preference` one-hot encoded as an extra input. Does the recall gap between women and
# men shrink or grow? Even if it shrank, what would you need to justify using it, and who would decide?

# %% [markdown]
# ## Checkpoint
#
# 1. The screen does not use sex. Can its errors still differ by sex? <details><summary>Answer</summary>Yes: the
#    inputs (here, recorded prior falls) can differ in quality by group, and a proxy can carry the attribute.
#    </details>
# 2. Why report the counts next to each rate? <details><summary>Answer</summary>A rate on a few dozen people moves
#    a lot by chance; the counts show how much weight it can bear.</details>
# 3. Who signs the model card? <details><summary>Answer</summary>The analyst prepares it, someone else reviews it
#    (the quality and data protection officer), and the clinical director accepts the screen for use.</details>
#
# **Carry forward.** `artifacts/D4_3_model_card.md` (graded). The capstone asks for the same card on a new screen.
