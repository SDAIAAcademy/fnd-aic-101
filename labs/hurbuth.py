"""Shared helpers for the labs of How Machine Learning Works (SDAIA Academy, FND-AIC-101).

Hurbuth Home Care Company is fictional. Every file in data/, capstone/ and followup/ is synthetic and
written by prepare_data.py with a fixed seed. Clients are codes; nothing identifies a person.

    import hurbuth as U
    df = U.assessments()                 # 4,776 home assessments, 2023 to mid-2025
    train, val, test = U.split(df)       # the course's fixed split by date
    X, y = train[U.FEATURES], train[U.TARGET]

Environment variables:
    FNDAIC101_QUICK=1      quick mode (U.pick returns its small value); every lab is fast anyway
    FNDAIC101_STRICT=1     solution runs: a failed self-check stops the run
    FNDAIC101_WRITE_REF=1  the trainer's run that writes data/reference/ (the safety-net values)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):   # Windows consoles: print check marks
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
ART = LABS / "artifacts"            # what a lab writes: plots, tables, notes (git ignores it)

QUICK = os.environ.get("FNDAIC101_QUICK", "0") == "1"
STRICT = os.environ.get("FNDAIC101_STRICT", "0") == "1"
WRITE_REF = os.environ.get("FNDAIC101_WRITE_REF", "0") == "1"
SEED = 101

TARGET = "fell_6m"
FEATURES = ["age", "lives_alone", "n_medications", "prior_falls_12m", "walk_seconds", "vision_problem",
            "uses_walking_aid", "stairs_at_home", "diabetes"]
CATEGORICAL = ["sector"]
SENSITIVE = ["sex"]                  # never a model input; used to measure error rates by group
TRAIN_END, VAL_END = "2024-07-01", "2025-01-01"
COST_MISSED_FALL, COST_NEEDLESS_PROGRAMME = 6, 1   # cost units of the case (Lab 2.4)


# ---------------------------------------------------------------------------------------------------------------
# Small tools every lab uses
# ---------------------------------------------------------------------------------------------------------------

def pick(full, quick):
    """The full value, or the small one in quick mode (FNDAIC101_QUICK=1)."""
    return quick if QUICK else full


def check(ok, what: str) -> bool:
    """Print a self-check. In a solution run (FNDAIC101_STRICT=1) a failed check stops the run."""
    ok = bool(ok)
    print(("✅ " if ok else "❌ ") + what)
    if STRICT and not ok:
        raise AssertionError(what)
    return ok


def _to_json(value):
    if isinstance(value, pd.DataFrame):
        return {"__frame__": json.loads(value.to_json(orient="split"))}
    if isinstance(value, pd.Series):
        return {"__series__": json.loads(value.to_json(orient="split"))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    return value


def _from_json(value):
    if isinstance(value, dict) and "__frame__" in value:
        f = value["__frame__"]
        return pd.DataFrame(f["data"], index=f["index"], columns=f["columns"])
    if isinstance(value, dict) and "__series__" in value:
        s = value["__series__"]
        return pd.Series(s["data"], index=s["index"], name=s.get("name"))
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_json(v) for k, v in value.items()}
    return value


def keep(name: str, value) -> None:
    """In the trainer's reference run, store a result for the safety net of a later cell."""
    if WRITE_REF:
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(_to_json(value), indent=1) + "\n", encoding="utf-8", newline="\n")


def reference(name: str):
    """The reference result of an exercise, so the cells after it run when it is not done yet."""
    if STRICT:
        raise AssertionError(f"a solution run took the safety net for {name}")
    print(f"ℹ️  using the reference result for {name}: finish the exercise above to see your own")
    return _from_json(json.loads((REF / f"{name}.json").read_text(encoding="utf-8")))


def answer(name: str):
    """A stored answer key (no safety-net message); None when the trainer's run has not written it yet."""
    p = REF / f"{name}.json"
    return _from_json(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def save(name: str, text: str) -> Path:
    """Write a text artefact (a table, a note, a model card) to labs/artifacts/."""
    ART.mkdir(exist_ok=True)
    p = ART / name
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote artifacts/{name}")
    return p


def style():
    """Matplotlib with the course's defaults; no window opens when a notebook runs as a script."""
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def savefig(fig, name: str) -> None:
    ART.mkdir(exist_ok=True)
    fig.savefig(ART / name, dpi=110, bbox_inches="tight")
    print(f"wrote artifacts/{name}")


# ---------------------------------------------------------------------------------------------------------------
# The case data
# ---------------------------------------------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "assessment_date" in df:
        df["assessment_date"] = pd.to_datetime(df["assessment_date"])
    return df


def assessments() -> pd.DataFrame:
    """Every home assessment from January 2023 to June 2025, with its six-month outcome."""
    return _read(DATA / "assessments.csv")


def split(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The course's fixed split by date: training before July 2024, validation to December 2024, test 2025."""
    df = assessments() if df is None else df
    d = pd.to_datetime(df["assessment_date"])
    return (df[d < TRAIN_END].copy(), df[(d >= TRAIN_END) & (d < VAL_END)].copy(), df[d >= VAL_END].copy())


def ten_rows() -> pd.DataFrame:
    """The ten training rows of Day 1."""
    return _read(DATA / "ten_rows.csv")


def later_batch() -> pd.DataFrame:
    """Assessments of July to December 2025, including the newly opened Central sector."""
    return _read(DATA / "later_batch.csv")


def leaky_export() -> pd.DataFrame:
    """The export a former analyst's pipeline used (Lab 3.4)."""
    return _read(DATA / "leaky" / "assessments_export.csv")


def sensor_pilot() -> pd.DataFrame:
    """240 clients of a home-activity pilot and 400 activity columns (Lab 3.3)."""
    return pd.read_csv(DATA / "sensor_pilot.csv")


def scores(which: str) -> pd.DataFrame:
    """Scores of a fixed screen on the validation ('val') or test ('test') assessments."""
    return pd.read_csv(DATA / "scores" / f"{which}_scores.csv")


def admissions() -> pd.DataFrame:
    """The capstone's screen: unplanned hospital admission within 90 days."""
    return _read(LABS / "capstone" / "admissions.csv")


def pressure_injuries() -> pd.DataFrame:
    """The follow-up task's screen: a pressure injury within 90 days."""
    return _read(LABS / "followup" / "pressure_injury.csv")


def confusion_counts(y_true, flagged) -> dict[str, int]:
    """The four cells, named for a screen: caught fallers, needless flags, missed fallers, correct passes."""
    y, f = np.asarray(y_true).astype(bool), np.asarray(flagged).astype(bool)
    return {"tp": int((y & f).sum()), "fp": int((~y & f).sum()), "fn": int((y & ~f).sum()), "tn": int((~y & ~f).sum())}


def standard_screen(train: pd.DataFrame):
    """The course's standard fall screen: a scaled logistic regression on FEATURES, fitted on `train`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(train[FEATURES], train[TARGET])


def cheapest_threshold(y, score, c_fn: float = COST_MISSED_FALL, c_fp: float = COST_NEEDLESS_PROGRAMME) -> float:
    """The threshold from 0.02 to 0.60 with the lowest expected cost (the method of Lab 2.4)."""
    y, score = np.asarray(y), np.asarray(score)
    grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
    costs = [c_fn * ((score < t) & (y == 1)).sum() + c_fp * ((score >= t) & (y == 0)).sum() for t in grid]
    return float(grid[int(np.argmin(costs))])


def table(df: pd.DataFrame, digits: int = 3) -> str:
    """A DataFrame as a Markdown table for the artefacts."""
    return df.round(digits).to_markdown()
