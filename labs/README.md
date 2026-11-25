# How Machine Learning Works: labs

Every lab is a notebook that starts from files supplied in this repository, never from the output of an earlier
lab, and runs on a laptop CPU or a free hosted notebook in a few seconds to a minute. The data are synthetic,
written by `prepare_data.py` with seed 101, about a fictional home-care company, Hurbuth Home Care Company.

## Quick start

- **Hosted notebook (Colab):** open a notebook from the links below; its first cell clones this repository and
  installs `requirements-colab.txt`.
- **Locally:** clone the repository, then from `labs/` run `uv sync` and open the notebooks with Jupyter
  (`uv run --with jupyterlab jupyter lab`). On Windows, if the folder path is deep, set `UV_PROJECT_ENVIRONMENT`
  to a short folder first.
- Start with [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/00_setup_check.ipynb):
  it should print 4,776 assessments and two check marks.

## Before Day 1: guided pre-work (75 min)

[`00_prework.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/00_prework.ipynb):
the slope of a squared error, and the dataframe operations the labs use.

## The labs

Graded labs are marked **G** (five labs, 6% of the course mark each).

| Lab | Session | Notebook | Min |
|---|---|---|---|
| 1.1 | A screening rule by hand | [`D1_1_hand_rule`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D1_1_hand_rule.ipynb) | 35 |
| 1.2 | Two gradient steps on ten rows | [`D1_2_gradient_steps`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D1_2_gradient_steps.ipynb) | 35 |
| 1.3 **G** | Two models by hand, then in code | [`D1_3_two_models_by_hand`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D1_3_two_models_by_hand.ipynb) | 40 |
| 1.4 | Baseline, line and tree on a fixed split | [`D1_4_three_way_split`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D1_4_three_way_split.ipynb) | 40 |
| 2.1 **G** | Overfit on purpose | [`D2_1_overfit_depth`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D2_1_overfit_depth.ipynb) | 40 |
| 2.2 | Grouped folds and six curves | [`D2_2_cross_validation`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D2_2_cross_validation.ipynb) | 35 |
| 2.3 | Four cells by hand, then in code | [`D2_3_confusion_matrix`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D2_3_confusion_matrix.ipynb) | 35 |
| 2.4 **G** | A threshold from costs | [`D2_4_threshold`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D2_4_threshold.ipynb) | 40 |
| 3.1 | Visit minutes against two baselines | [`D3_1_regression_baselines`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D3_1_regression_baselines.ipynb) | 35 |
| 3.2 | A column transformer for the screen | [`D3_2_column_transformer`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D3_2_column_transformer.ipynb) | 35 |
| 3.3 | Selection before the split | [`D3_3_pipeline_leak`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D3_3_pipeline_leak.ipynb) | 40 |
| 3.4 **G** | Three seeded leaks | [`D3_4_seeded_leaks`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D3_4_seeded_leaks.ipynb) | 40 |
| 4.1 | Three families and a baseline | [`D4_1_three_families`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D4_1_three_families.ipynb) | 40 |
| 4.2 | A later batch from a new sector | [`D4_2_later_batch`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D4_2_later_batch.ipynb) | 35 |
| 4.3 **G** | Subgroup error rates and a model card | [`D4_3_subgroups_model_card`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D4_3_subgroups_model_card.ipynb) | 40 |
| 4.4 | The capstone starter | [`D4_4_capstone_start`](https://colab.research.google.com/github/SDAIAAcademy/fnd-aic-101/blob/main/labs/D4_4_capstone_start.ipynb) | 35 |

If an exercise is not finished, the cells after it use a stored reference result (and say so), so the rest of the
notebook still runs. Worked solutions are in `solutions/`.

## Guided self-study between days (20 min each)

After each of Days 1 to 3: finish any lab you did not finish, then answer that day's practice questions on the
course page without an assistant. If time remains, the stretch of the day's last graded lab (1.3, 2.4, 3.4).

## Capstone and follow-up

- [`capstone/README.md`](capstone/README.md): the brief, the data dictionary, the costs and the rubric of the
  admission screen. Start from Lab 4.4.
- [`model_card_template.md`](model_card_template.md): the one-page model card used in Lab 4.3 and the capstone.
- [`followup/README.md`](followup/README.md): the follow-up task on the pressure-injury screen.

## For trainers

- The notebooks are generated from the percent-format sources in `_src/` by `_tools/py2nb.py`
  (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `solutions/`. Edit the
  sources, not the notebooks.
- `uv run python prepare_data.py` rewrites every data file from seed 101. `FNDAIC101_WRITE_REF=1` with a solution
  run rewrites the reference results in `data/reference/`.
- [`solutions/answer_key.md`](solutions/answer_key.md): the planted problems of each lab and the numbers a correct
  run prints.
- Every lab writes its plots and notes to `labs/artifacts/`, which git ignores.
