# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.6"]
# ///
"""The parallel examples the slides show before each lab, computed here so that every number on a
slide comes from a run.

    cd labs
    uv run _tools/teaching_examples.py

The slides teach on a different team of the same fictional company, the wound-care nurses, with their
own synthetic visits (seed 7, made here and never saved), their own label (did a dressing visit run
over 40 minutes) and their own numbers. The fall-risk numbers of the labs appear on slides only in the
debrief after each lab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 7


def wound_visits(n: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    patient = rng.integers(0, 900, n)
    area = np.round(rng.gamma(2.0, 6.0, n), 1)                 # wound area, square centimetres
    layers = rng.integers(1, 5, n)                              # dressing layers
    diabetic = (rng.random(n) < 0.35).astype(int)
    age_band = rng.choice(["65-74", "75-84", "85+"], n, p=[0.45, 0.4, 0.15])
    minutes = 14 + 0.9 * area + 3.0 * layers + 5 * diabetic + rng.normal(0, 6, n)
    df = pd.DataFrame({"patient": patient, "area_cm2": area, "layers": layers, "diabetic": diabetic,
                       "age_band": age_band, "minutes": np.round(minutes).astype(int)})
    df["over_40"] = (df.minutes > 40).astype(int)
    df["month"] = np.sort(rng.integers(0, 24, n))
    return df


def main() -> None:
    w = wound_visits()
    print("== wound-care visits:", len(w), "visits; share over 40 minutes", round(w.over_40.mean(), 3))

    # Day 1, session 1: a hand rule on six visits
    six = pd.concat([w[w.over_40 == 1].head(3), w[w.over_40 == 0].head(3)]).sort_index()[["area_cm2", "layers", "diabetic", "over_40"]]
    print("\n== D1 S1 six visits\n", six.to_string(index=False))
    flag = six.area_cm2 >= 15
    print("rule area >= 15 flags", int(flag.sum()), "caught", int((flag & (six.over_40 == 1)).sum()),
          "of", int(six.over_40.sum()))

    # Day 1, session 2: a line for minutes from area on five visits, loss and one gradient step
    five = pd.DataFrame({"area_cm2": [4.0, 8.0, 12.0, 16.0, 20.0], "minutes": [20, 26, 27, 33, 39]})
    x, y = five.area_cm2.to_numpy(), five.minutes.to_numpy(float)
    for wv, bv in [(1.0, 15.0), (1.2, 15.0)]:
        print(f"\n== D1 S2 line w={wv} b={bv}: errors {[float(v) for v in np.round(wv * x + bv - y, 1)]}, "
              f"mse {np.mean((wv * x + bv - y) ** 2):.2f}")
    e = 1.0 * x + 15 - y
    gw, gb = 2 * np.mean(e * x), 2 * np.mean(e)
    print(f"gradient at w=1, b=15: slope_w {gw:.2f}, slope_b {gb:.2f}; step 0.002 -> w {1 - 0.002 * gw:.4f}, "
          f"b {15 - 0.002 * gb:.4f}, mse {np.mean(((1 - 0.002 * gw) * x + 15 - 0.002 * gb - y) ** 2):.2f}")
    wl, bl = np.polyfit(x, y, 1)
    print(f"least squares on the five: w {wl:.3f}, b {bl:.3f}, mse {np.mean((wl * x + bl - y) ** 2):.3f}")

    # Day 1, session 3: neighbours on two features, and the unit trap
    pts = pd.DataFrame({"area_cm2": [5, 6, 18, 20, 9, 22], "layers": [1, 3, 2, 4, 4, 1], "over_40": [0, 0, 1, 1, 1, 0]})
    new = (10, 1)
    d = np.sqrt((pts.area_cm2 - new[0]) ** 2 + (pts.layers - new[1]) ** 2)
    print("\n== D1 S3 points\n", pts.assign(distance=d.round(2)).to_string(index=False))
    near = d.sort_values(kind="stable").index[:3]
    print("3-NN vote for (10 cm2, 1 layer):", int(pts.loc[near, "over_40"].mean() > 0.5), "neighbours", list(near))
    d2 = np.sqrt((pts.area_cm2 / 10000 - new[0] / 10000) ** 2 + (pts.layers - new[1]) ** 2)
    near2 = d2.sort_values(kind="stable").index[:3]
    print("area in square metres: nearest three", list(near2), "vote", int(pts.loc[near2, "over_40"].mean() > 0.5))

    # Day 1, session 4: split by month, baseline against a model
    tr, va, te = w[w.month < 14], w[(w.month >= 14) & (w.month < 19)], w[w.month >= 19]
    F = ["area_cm2", "layers", "diabetic"]
    w2 = w.assign(over_50=(w.minutes > 50).astype(int))
    tr2, va2 = w2[w2.month < 14], w2[(w2.month >= 14) & (w2.month < 19)]
    base = DummyClassifier(strategy="most_frequent").fit(tr2[F], tr2.over_50)
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr2[F], tr2.over_50)
    print(f"\n== D1 S4 label over 50 minutes: base rate {w2.over_50.mean():.3f}; rows train {len(tr2)}, "
          f"validation {len(va2)}, test {len(w2) - len(tr2) - len(va2)}")
    print(f"validation: baseline accuracy {accuracy_score(va2.over_50, base.predict(va2[F])):.3f}, "
          f"logistic accuracy {accuracy_score(va2.over_50, lr.predict(va2[F])):.3f}, "
          f"AUC {roc_auc_score(va2.over_50, lr.predict_proba(va2[F])[:, 1]):.3f}")

    # Day 2, session 1: depth curve on the wound data (label over 50 minutes)
    print("\n== D2 S1 depth curve (accuracy, label over 50 minutes)")
    for depth in [1, 2, 3, 4, 6, 8, 12, None]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(tr2[F + ["month"]], tr2.over_50)
        print(f"depth {depth}: train {t.score(tr2[F + ['month']], tr2.over_50):.3f}, "
              f"validation {t.score(va2[F + ['month']], va2.over_50):.3f}")

    # Day 2, session 2: grouped folds by patient
    g = GroupKFold(n_splits=5)
    sc = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED), tr2[F], tr2.over_50,
                         groups=tr2.patient, cv=g, scoring="roc_auc")
    print("\n== D2 S2 grouped 5-fold AUC:", [float(v) for v in np.round(sc, 3)], "mean", round(sc.mean(), 3), "sd", round(sc.std(), 3))

    # Day 2, session 3: a confusion matrix at 0.5 and at 0.2
    s = lr.predict_proba(va2[F])[:, 1]
    for th in (0.5, 0.2):
        f = s >= th
        yv = va2.over_50.to_numpy() == 1
        tp, fp, fn, tn = int((f & yv).sum()), int((f & ~yv).sum()), int((~f & yv).sum()), int((~f & ~yv).sum())
        print(f"\n== D2 S3 threshold {th}: tp {tp} fp {fp} fn {fn} tn {tn}; precision {tp / max(tp + fp, 1):.3f} "
              f"recall {tp / (tp + fn):.3f} specificity {tn / (tn + fp):.3f} accuracy {(tp + tn) / len(f):.3f}")

    # Day 2, session 4: expected cost with costs 4 (missed) and 1 (needless) per 1,000 visits
    print("\n== D2 S4 cost per 1,000 visits, missed 4, needless 1")
    for th in (0.1, 0.2, 0.3, 0.5):
        f = s >= th
        yv = va2.over_50.to_numpy() == 1
        cost = 4 * (~f & yv).sum() + 1 * (f & ~yv).sum()
        print(f"threshold {th}: cost per 1,000 {1000 * cost / len(f):.0f}, flagged {f.mean():.3f}")
    print("cost-ratio rule: 1 / (1 + 4) =", 1 / 5)

    # Day 3, session 1: regression against naive baselines
    reg = LinearRegression().fit(tr[F], tr.minutes)
    print(f"\n== D3 S1 minutes: MAE mean baseline {mean_absolute_error(va.minutes, np.full(len(va), tr.minutes.mean())):.2f}, "
          f"median baseline {mean_absolute_error(va.minutes, np.full(len(va), tr.minutes.median())):.2f}, "
          f"linear {mean_absolute_error(va.minutes, reg.predict(va[F])):.2f}; RMSE linear "
          f"{mean_squared_error(va.minutes, reg.predict(va[F])) ** 0.5:.2f}")
    five_err = np.array([3, -2, 10, -1, 0])
    print(f"five errors {[int(v) for v in five_err]}: MAE {np.abs(five_err).mean():.2f}, RMSE {np.sqrt((five_err ** 2).mean()):.2f}")

    # Day 3, session 2: scaling and a neighbour model
    raw = KNeighborsClassifier(5).fit(tr2[["area_cm2", "layers"]].assign(area_cm2=tr2.area_cm2 * 100), tr2.over_50)
    scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(tr2[["area_cm2", "layers"]], tr2.over_50)
    print(f"\n== D3 S2 5-NN AUC, area in mm2 unscaled "
          f"{roc_auc_score(va2.over_50, raw.predict_proba(va2[['area_cm2', 'layers']].assign(area_cm2=va2.area_cm2 * 100))[:, 1]):.3f}, "
          f"scaled {roc_auc_score(va2.over_50, scaled.predict_proba(va2[['area_cm2', 'layers']])[:, 1]):.3f}")

    # Day 3, session 3: feature selection on random columns (a different size from the lab's pilot)
    rng = np.random.default_rng(SEED + 1)
    Xn, yn = rng.normal(size=(150, 600)), rng.integers(0, 2, 150)
    leaky = cross_val_score(LogisticRegression(max_iter=1000), SelectKBest(f_classif, k=10).fit_transform(Xn, yn), yn,
                            cv=5, scoring="roc_auc").mean()
    clean = cross_val_score(make_pipeline(SelectKBest(f_classif, k=10), LogisticRegression(max_iter=1000)), Xn, yn,
                            cv=5, scoring="roc_auc").mean()
    print(f"\n== D3 S3 150 rows, 600 random columns, keep 10: selection first AUC {leaky:.3f}, inside pipeline {clean:.3f}")

    # Day 4, session 3: error rates by age band at one threshold
    f = s >= 0.2
    print("\n== D4 S3 recall by age band at 0.2 (label over 50 minutes)")
    for band in ["65-74", "75-84", "85+"]:
        m = (va2.age_band == band).to_numpy()
        yv = va2.over_50.to_numpy()[m] == 1
        print(f"{band}: visits {m.sum()}, over 50 {yv.sum()}, recall {(f[m] & yv).sum() / max(yv.sum(), 1):.3f}")


if __name__ == "__main__":
    main()
