<!-- academy:title -->
# How Machine Learning Works

كيف يعمل تعلم الآلة

Data & AI Foundations Track · SDAIA Academy · FND-AIC-101 · Level 100

REPLACE: one line under the title: what participants can do after the course

Participants work out how supervised learning turns labelled examples into predictions, by hand first and then in code. They fit a nearest-neighbour rule and a linear model on paper, make a model overfit on purpose, choose a decision threshold from error costs and base rates, and hunt leaks seeded in a pipeline. Labs run on a laptop with small public or synthetic tabular data, and the capstone is an honest modelling report with a model card, defended in person.

**Format:** Intensive: 4 teaching days and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/fnd-aic-101/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Explain how a supervised learner fits a prediction rule by minimising a loss, given a ten-row table and a one-feature linear model | Awareness | a hand-worked loss and two gradient steps, with a three-sentence explanation |
| LO2 | Distinguish training, validation and test data, and identify overfitting and underfitting from plotted curves, given six complexity and learning curves | Awareness | a diagnosis and remedy per curve, at least five of six correct |
| LO3 | Identify target, duplicate, preprocessing and temporal leakage in modelling pipelines, given eight short pipeline descriptions | Awareness | each leak named with its fix, at least six of eight correct |
| LO4 | Implement a train, validate and test workflow comparing a baseline, a linear model and a decision tree, given a tabular dataset and a starter notebook | Applied | a notebook that runs end to end with a fixed seed and reports held-out metrics |
| LO5 | Compute confusion-matrix metrics and choose a threshold from error costs and the base rate, given test-set scores and a cost per error type | Applied | a metrics table and a justified threshold |
| LO6 | Draft a one-page model card with intended use, data, metrics by subgroup and known limits, given the participant's capstone model | Applied | a model card reviewed in the defence |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1; REPLACE: the title of Session 2; REPLACE: the title of Session 3; REPLACE: the title of Session 4 | [`slides/day1.html`](slides/day1.html) |
| Day 2 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1; REPLACE: the title of Session 2; REPLACE: the title of Session 3; REPLACE: the title of Session 4 | [`slides/day2.html`](slides/day2.html) |
| Day 3 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1; REPLACE: the title of Session 2; REPLACE: the title of Session 3; REPLACE: the title of Session 4 | [`slides/day3.html`](slides/day3.html) |
| Day 4 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1; REPLACE: the title of Session 2; REPLACE: the title of Session 3; REPLACE: the title of Session 4 | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | REPLACE: what participants build in lab 1.1 | [`D1_1.ipynb`](labs/D1_1.ipynb) |
| 1.2 | Day 1 · Session 2 | REPLACE: what participants build in lab 1.2 | [`D1_2.ipynb`](labs/D1_2.ipynb) |
| 1.3 | Day 1 · Session 3 | REPLACE: what participants build in lab 1.3 | [`D1_3.ipynb`](labs/D1_3.ipynb) |
| 1.4 | Day 1 · Session 4 | REPLACE: what participants build in lab 1.4 | [`D1_4.ipynb`](labs/D1_4.ipynb) |
| 2.1 | Day 2 · Session 1 | REPLACE: what participants build in lab 2.1 | [`D2_1.ipynb`](labs/D2_1.ipynb) |
| 2.2 | Day 2 · Session 2 | REPLACE: what participants build in lab 2.2 | [`D2_2.ipynb`](labs/D2_2.ipynb) |
| 2.3 | Day 2 · Session 3 | REPLACE: what participants build in lab 2.3 | [`D2_3.ipynb`](labs/D2_3.ipynb) |
| 2.4 | Day 2 · Session 4 | REPLACE: what participants build in lab 2.4 | [`D2_4.ipynb`](labs/D2_4.ipynb) |
| 3.1 | Day 3 · Session 1 | REPLACE: what participants build in lab 3.1 | [`D3_1.ipynb`](labs/D3_1.ipynb) |
| 3.2 | Day 3 · Session 2 | REPLACE: what participants build in lab 3.2 | [`D3_2.ipynb`](labs/D3_2.ipynb) |
| 3.3 | Day 3 · Session 3 | REPLACE: what participants build in lab 3.3 | [`D3_3.ipynb`](labs/D3_3.ipynb) |
| 3.4 | Day 3 · Session 4 | REPLACE: what participants build in lab 3.4 | [`D3_4.ipynb`](labs/D3_4.ipynb) |
| 4.1 | Day 4 · Session 1 | REPLACE: what participants build in lab 4.1 | [`D4_1.ipynb`](labs/D4_1.ipynb) |
| 4.2 | Day 4 · Session 2 | REPLACE: what participants build in lab 4.2 | [`D4_2.ipynb`](labs/D4_2.ipynb) |
| 4.3 | Day 4 · Session 3 | REPLACE: what participants build in lab 4.3 | [`D4_3.ipynb`](labs/D4_3.ipynb) |
| 4.4 | Day 4 · Session 4 | REPLACE: what participants build in lab 4.4 | [`D4_4.ipynb`](labs/D4_4.ipynb) |
<!-- /academy:labs -->

## Capstone

A modelling report on a provided public or synthetic dataset, with a baseline, two models, a threshold decision and a model card, defended in ten minutes; the follow-up task repeats it on a second dataset

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- FND-PRG-101 Python Programming for Data and AI
- Writes short Python functions and loops; a passed placement exercise is accepted instead of FND-PRG-101
- Uses school algebra; the derivative of a squared error is taught in the pre-work

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Notebook environment:** JupyterLab, a free hosted notebook service
- **Machine learning library:** scikit-learn, tidymodels in R
- **Dataframes and plotting:** pandas with Matplotlib, Polars with seaborn
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: REPLACE: the address of the official text
