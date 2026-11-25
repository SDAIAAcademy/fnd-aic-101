# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Write the synthetic data of Hurbuth Home Care Company, the fictional case of FND-AIC-101.

    cd labs
    uv run prepare_data.py

Fixed seed, same files on every run. Everything is invented: the company, its unnamed city, its
five sectors, its clients (codes only, no names, no identifiers of any real kind) and every number.
The numbers are not clinical evidence about falls.

How the data is made. Each client gets a stable profile and a hidden frailty. Falls are simulated as
events in time, month by month, from a hazard that depends on the profile and the frailty. An
assessment on a date then reads the world as a nurse would:

- `fell_6m`, the label: did the client fall in the 182 days after the assessment;
- `prior_falls_12m`: falls in the 365 days before, as recorded. Falls of men are under-recorded on
  purpose (each is recorded with probability 0.35, against 0.95 for women): the planted cause of the
  subgroup gap that Lab 4.3 measures;
- `visit_minutes`: the length of the nurse's visit, the regression target of Labs 1.2, 1.3 and 3.1.

Files written to labs/data/ (and the capstone and follow-up folders):

  assessments.csv          4,000-odd assessments, Jan 2023 to Jun 2025, sectors North to West
  later_batch.csv          Jul to Dec 2025, with the new Central sector (Lab 4.2)
  ten_rows.csv             the ten-row table of Day 1 (Labs 1.1 to 1.3)
  sensor_pilot.csv         240 clients and 400 activity columns that carry no signal (Lab 3.3)
  leaky/assessments_export.csv   the export with three seeded leaks (Lab 3.4)
  scores/val_scores.csv, scores/test_scores.csv   scores of a fixed screen (Labs 2.3 and 2.4)
  capstone/admissions.csv  the 90-day unplanned admission screen (capstone)
  followup/pressure_injury.csv   the pressure-injury screen (follow-up task)

Planted problems, the lab that finds each, and the expected numbers are written by
`labs/_tools/make_reference.py` into labs/solutions/answer_key.md and labs/data/reference/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 101
LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
SECTORS = ["North", "South", "East", "West"]
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-07-01")
MALE_RECORDED, FEMALE_RECORDED = 0.35, 0.95


def clients(rng: np.random.Generator, n: int, prefix: str, sectors: list[str], age_shift: float = 0.0) -> pd.DataFrame:
    sex = rng.choice(["F", "M"], size=n, p=[0.58, 0.42])
    age0 = np.clip(66 + rng.gamma(2.2, 4.6, n) + age_shift, 65, 98)
    female = sex == "F"
    pref = np.where(
        female,
        rng.choice(["female_nurse", "no_preference"], size=n, p=[0.88, 0.12]),
        rng.choice(["male_nurse", "no_preference", "female_nurse"], size=n, p=[0.52, 0.42, 0.06]),
    )
    df = pd.DataFrame({
        "client_id": [f"{prefix}{i:04d}" for i in range(1, n + 1)],
        "sector": rng.choice(sectors, size=n),
        "sex": sex,
        "age0": age0,
        "nurse_preference": pref,
        "lives_alone": (rng.random(n) < np.where(female, 0.27, 0.15)).astype(int),
        "diabetes": (rng.random(n) < 0.32).astype(int),
        "stairs_at_home": (rng.random(n) < 0.44).astype(int),
        "frailty": rng.normal(0, 1.15, n),
    })
    df["vision_problem"] = (rng.random(n) < 0.12 + 0.012 * (df.age0 - 65)).astype(int)
    df["uses_walking_aid"] = (rng.random(n) < 0.10 + 0.018 * (df.age0 - 65) + 0.15 * (df.frailty > 0.5)).astype(int)
    df["meds_base"] = rng.poisson(2.2 + 0.07 * (df.age0 - 65) + 1.8 * df.diabetes)
    df["bmi"] = np.round(rng.normal(27.5, 4.6, n), 1)
    return df


def hazard(c: pd.DataFrame) -> np.ndarray:
    """Falls per month for each client (before age changes); calibrated to a 6-month rate near 15%."""
    lin = (-4.6 + 0.035 * (c.age0 - 75) + 0.45 * c.uses_walking_aid + 0.55 * c.vision_problem
           + 0.30 * c.lives_alone + 0.20 * c.stairs_at_home + 0.06 * (c.meds_base - 4) + c.frailty)
    return np.exp(np.minimum(lin.to_numpy(), -1.9))   # at most about one fall in seven months


def simulate_falls(rng: np.random.Generator, c: pd.DataFrame) -> dict[str, np.ndarray]:
    lam = hazard(c)
    months = pd.date_range(START, END, freq="MS")
    falls: dict[str, list[pd.Timestamp]] = {cid: [] for cid in c.client_id}
    for m in months[:-1]:
        k = rng.poisson(lam)
        for idx in np.flatnonzero(k):
            for _ in range(k[idx]):
                falls[c.client_id.iat[idx]].append(m + pd.Timedelta(days=int(rng.integers(0, 28))))
    return {cid: np.array(sorted(v), dtype="datetime64[ns]") for cid, v in falls.items()}


def assess(rng: np.random.Generator, c: pd.DataFrame, falls: dict, first: tuple[str, str], last: str,
           walk_metres: dict[str, float] | None = None, repeats: bool = True) -> pd.DataFrame:
    rows = []
    lo, hi = pd.Timestamp(first[0]), pd.Timestamp(first[1])
    last = pd.Timestamp(last)
    span = (hi - lo).days
    for r in c.itertuples(index=False):
        d = lo + pd.Timedelta(days=int(rng.integers(0, span)))
        while d <= last:
            rows.append((r, d))
            if not repeats:
                break
            d = d + pd.Timedelta(days=int(rng.integers(240, 420)))
    out = []
    for r, d in rows:
        f = falls[r.client_id]
        before = f[(f >= np.datetime64(d - pd.Timedelta(days=365))) & (f < np.datetime64(d))]
        after = f[(f > np.datetime64(d)) & (f <= np.datetime64(d + pd.Timedelta(days=182)))]
        p_rec = MALE_RECORDED if r.sex == "M" else FEMALE_RECORDED
        recorded = int((rng.random(len(before)) < p_rec).sum())
        age = r.age0 + (d - pd.Timestamp("2023-01-01")).days / 365.25
        meds = int(np.clip(r.meds_base + rng.integers(-1, 2), 0, 14))
        walk4 = 3.6 + 0.07 * (age - 65) + 1.6 * r.uses_walking_aid + 0.6 * max(r.frailty, 0) + rng.gamma(2.0, 0.35)
        metres = (walk_metres or {}).get(r.sector, 4.0)
        visit = 18 + 3.0 * meds + 6 * r.uses_walking_aid + 5 * r.diabetes + 4 * r.lives_alone + rng.normal(0, 5.5)
        out.append({
            "client_id": r.client_id, "assessment_date": d.date().isoformat(), "sector": r.sector,
            "sex": r.sex, "age": int(age), "lives_alone": r.lives_alone, "n_medications": meds,
            "prior_falls_12m": recorded, "walk_seconds": round(walk4 * metres / 4.0, 1),
            "vision_problem": r.vision_problem, "uses_walking_aid": r.uses_walking_aid,
            "stairs_at_home": r.stairs_at_home, "diabetes": r.diabetes,
            "bmi": r.bmi if rng.random() > (0.12 if r.lives_alone else 0.05) else np.nan,
            "nurse_preference": r.nurse_preference, "visit_minutes": int(round(max(visit, 10))),
            "fell_6m": int(len(after) > 0), "_true_prior": len(before), "_falls_after": len(after),
        })
    df = pd.DataFrame(out).sort_values(["assessment_date", "client_id"]).reset_index(drop=True)
    return df


def add_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df.insert(0, "assessment_id", [f"{prefix}{i:05d}" for i in range(1, len(df) + 1)])
    return df


PUBLIC = ["assessment_id", "client_id", "assessment_date", "sector", "sex", "age", "lives_alone", "n_medications",
          "prior_falls_12m", "walk_seconds", "vision_problem", "uses_walking_aid", "stairs_at_home", "diabetes",
          "bmi", "nurse_preference", "visit_minutes", "fell_6m"]


def ten_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ten training rows, one per number of medicines from 1 to 10, chosen by rule (no hand editing)."""
    pool = df[(df.assessment_date < "2024-07-01") & df.bmi.notna()].copy()
    picks, used = [], set()
    wanted = [(1, 0), (2, 0), (3, 0), (4, 1), (5, 0), (6, 1), (7, 0), (8, 1), (9, 0), (10, 1)]
    for meds, fell in wanted:
        cand = pool[(pool.n_medications == meds) & ~pool.client_id.isin(used)]
        # fallers with a recorded prior fall and non-fallers without one, so the neighbour rule has
        # something to find; relaxed step by step when the pool has no such row
        strict = cand[(cand.fell_6m == fell) & ((cand.prior_falls_12m >= 1) == bool(fell))]
        cand = strict if len(strict) else cand[cand.fell_6m == fell] if len(cand[cand.fell_6m == fell]) else cand
        row = cand.iloc[0]
        used.add(row.client_id)
        picks.append(row)
    t = pd.DataFrame(picks)[["client_id", "n_medications", "visit_minutes", "prior_falls_12m", "walk_seconds", "fell_6m"]]
    return t.reset_index(drop=True)


def leaky_export(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """The export a former analyst used: three seeded leaks (see the answer key)."""
    x = df[PUBLIC].copy()
    # 1 · target leak: a physiotherapy referral that is made after a fall in the follow-up window
    after_fall = (x.fell_6m == 1) & (rng.random(len(x)) < 0.82)
    routine = (x.fell_6m == 0) & (rng.random(len(x)) < 0.05)
    x["physio_referral"] = (after_fall | routine).astype(int)
    # 3 · temporal leak: total recorded falls of the client over every assessment in the file
    x["client_falls_total"] = x.groupby("client_id").prior_falls_12m.transform("sum")
    # 2 · duplicate leak: a sync fault saved 12% of assessments twice
    dup = x.sample(frac=0.12, random_state=SEED)
    x = pd.concat([x, dup]).sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    return x


def sensor_pilot(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    pilot = df[df.assessment_date < "2024-07-01"].drop_duplicates("client_id").sample(240, random_state=SEED)
    noise = np.round(rng.normal(0, 1, (240, 400)), 2)
    cols = {f"activity_{j:03d}": noise[:, j] for j in range(400)}
    out = pd.DataFrame({"client_id": pilot.client_id.to_numpy(), "fell_6m": pilot.fell_6m.to_numpy(), **cols})
    return out


def scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores of a fixed screen (a logistic regression on four features, fitted on the training period).

    Written with numpy only so the generator needs no machine-learning library: a few hundred steps of
    gradient descent on the log loss, the same idea participants meet on Day 1.
    """
    feats = ["prior_falls_12m", "walk_seconds", "uses_walking_aid", "vision_problem", "lives_alone", "n_medications"]
    tr = df[df.assessment_date < "2024-07-01"]
    mu, sd = tr[feats].mean(), tr[feats].std()
    X = ((tr[feats] - mu) / sd).to_numpy()
    y = tr.fell_6m.to_numpy()
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= 0.1 * X.T @ (p - y) / len(y)
        b -= 0.1 * float(np.mean(p - y))
    out = []
    for lo, hi in [("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01")]:
        part = df[(df.assessment_date >= lo) & (df.assessment_date < hi)]
        z = ((part[feats] - mu) / sd).to_numpy() @ w + b
        out.append(pd.DataFrame({"assessment_id": part.assessment_id.to_numpy(), "sex": part.sex.to_numpy(),
                                 "score": np.round(1 / (1 + np.exp(-z)), 3), "fell_6m": part.fell_6m.to_numpy()}))
    return out[0], out[1]


def side_screen(rng: np.random.Generator, n: int, prefix: str, kind: str) -> pd.DataFrame:
    """A smaller synthetic screen of the same company: 'admission' (capstone) or 'pressure' (follow-up)."""
    c = clients(rng, n, prefix, SECTORS)
    d0 = pd.Timestamp("2023-01-01")
    dates = [d0 + pd.Timedelta(days=int(x)) for x in rng.integers(0, 912, n)]
    age = np.clip(c.age0.to_numpy() + rng.uniform(0, 2.5, n), 65, 99).astype(int)
    meds = np.clip(c.meds_base.to_numpy() + rng.integers(-1, 2, n), 0, 14)
    if kind == "admission":
        er_visits = rng.poisson(0.25 + 0.6 * (c.frailty.to_numpy() > 0.4) + 0.2 * c.diabetes.to_numpy())
        heart = (rng.random(n) < 0.22).astype(int)
        lin = (-3.1 + 0.03 * (age - 75) + 0.55 * er_visits + 0.6 * heart + 0.35 * c.diabetes.to_numpy()
               + 0.07 * (meds - 4) + 0.8 * c.frailty.to_numpy() + 0.25 * c.lives_alone.to_numpy())
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        recorded_er = np.where(c.sex.to_numpy() == "M", rng.binomial(er_visits, 0.6), rng.binomial(er_visits, 0.95))
        df = pd.DataFrame({"record_id": [f"H{i:05d}" for i in range(1, n + 1)], "client_id": c.client_id,
                           "assessment_date": [x.date().isoformat() for x in dates], "sector": c.sector,
                           "sex": c.sex, "age": age, "lives_alone": c.lives_alone, "n_medications": meds,
                           "er_visits_12m": recorded_er, "heart_failure": heart, "diabetes": c.diabetes,
                           "uses_walking_aid": c.uses_walking_aid, "nurse_preference": c.nurse_preference,
                           "bmi": np.where(rng.random(n) < 0.06, np.nan, c.bmi)})
        # planted: a field filled in after the outcome, for the participant to catch and drop
        df["discharge_letter_received"] = np.where(y == 1, (rng.random(n) < 0.9).astype(int), (rng.random(n) < 0.03).astype(int))
        df["admitted_90d"] = y
    else:
        mobility = np.clip(np.round(4 - 1.2 * c.uses_walking_aid.to_numpy() - 0.8 * (c.frailty.to_numpy() > 0.5)
                                    - 0.03 * (age - 75) + rng.normal(0, 0.7, n)), 1, 4).astype(int)
        incont = (rng.random(n) < 0.18 + 0.01 * (age - 75)).astype(int)
        lin = (-3.6 + 0.7 * (4 - mobility) + 0.8 * incont + 0.04 * (age - 75) - 0.08 * (c.bmi.to_numpy() - 27)
               + 0.6 * c.frailty.to_numpy())
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        df = pd.DataFrame({"record_id": [f"P{i:05d}" for i in range(1, n + 1)], "client_id": c.client_id,
                           "assessment_date": [x.date().isoformat() for x in dates], "sector": c.sector,
                           "sex": c.sex, "age": age, "lives_alone": c.lives_alone, "mobility_score": mobility,
                           "incontinence": incont, "diabetes": c.diabetes, "nurse_preference": c.nurse_preference,
                           "bmi": np.where(rng.random(n) < 0.06, np.nan, c.bmi)})
        df["wound_dressing_ordered"] = np.where(y == 1, (rng.random(n) < 0.85).astype(int), (rng.random(n) < 0.04).astype(int))
        df["pressure_injury_90d"] = y
    return df.sort_values("assessment_date").reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    (DATA / "leaky").mkdir(parents=True, exist_ok=True)
    (DATA / "scores").mkdir(exist_ok=True)
    (LABS / "capstone").mkdir(exist_ok=True)
    (LABS / "followup").mkdir(exist_ok=True)

    base = clients(rng, 2500, "C", SECTORS)
    central = clients(rng, 420, "N", ["Central"], age_shift=3.0)
    everyone = pd.concat([base, central], ignore_index=True)
    falls = simulate_falls(rng, everyone)

    main_df = add_ids(assess(rng, base, falls, ("2023-01-01", "2025-06-30"), "2025-06-30"), "A")
    later_old = assess(rng, base.sample(700, random_state=SEED), falls, ("2025-07-01", "2025-12-31"), "2025-12-31",
                       repeats=False)
    later_new = assess(rng, central, falls, ("2025-07-01", "2025-12-31"), "2025-12-31",
                       walk_metres={"Central": 6.0}, repeats=False)
    later = add_ids(pd.concat([later_old, later_new]).sort_values(["assessment_date", "client_id"])
                    .reset_index(drop=True), "B")

    main_df[PUBLIC].to_csv(DATA / "assessments.csv", index=False, lineterminator="\n")
    later[PUBLIC].to_csv(DATA / "later_batch.csv", index=False, lineterminator="\n")
    ten_rows(main_df).to_csv(DATA / "ten_rows.csv", index=False, lineterminator="\n")
    sensor_pilot(rng, main_df).to_csv(DATA / "sensor_pilot.csv", index=False, lineterminator="\n")
    leaky_export(rng, main_df).to_csv(DATA / "leaky" / "assessments_export.csv", index=False, lineterminator="\n")
    val, test = scores(main_df)
    val.to_csv(DATA / "scores" / "val_scores.csv", index=False, lineterminator="\n")
    test.to_csv(DATA / "scores" / "test_scores.csv", index=False, lineterminator="\n")
    side_screen(rng, 3000, "K", "admission").to_csv(LABS / "capstone" / "admissions.csv", index=False, lineterminator="\n")
    side_screen(rng, 2400, "P", "pressure").to_csv(LABS / "followup" / "pressure_injury.csv", index=False, lineterminator="\n")

    # A private check for the author: the recorded share of true prior falls by sex
    m = main_df
    for s in ("F", "M"):
        part = m[m.sex == s]
        print(f"sex {s}: {len(part)} assessments, base rate {part.fell_6m.mean():.3f}, "
              f"recorded/true prior falls {part.prior_falls_12m.sum()}/{part._true_prior.sum()}")
    print(f"assessments {len(m)} of {m.client_id.nunique()} clients, base rate {m.fell_6m.mean():.3f}; "
          f"later batch {len(later)}, base rate {later.fell_6m.mean():.3f}")


if __name__ == "__main__":
    main()
