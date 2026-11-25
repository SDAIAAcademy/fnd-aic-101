<!-- academy:title -->
# How Machine Learning Works

كيف يعمل تعلم الآلة

Data & AI Foundations Track · SDAIA Academy · FND-AIC-101 · Level 100

Work out by hand and then in code how a model learns from labelled examples, test it honestly on held-out data, set its threshold from error costs, catch leaks and write down its limits

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
| Day 1 | Learning from examples | Features, labels and prediction rules; Loss and gradient descent; Nearest neighbours and linear models by hand; Training, validation and test sets | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Generalisation and measuring | Model complexity, overfitting and regularisation; Cross-validation and learning curves; Confusion matrix, precision and recall; Thresholds, error costs and base rates | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Data and leakage | Regression errors against a naive baseline; Encoding, scaling and missing values; Pipelines fitted on training data only; Target, duplicate and temporal leakage | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Model families and limits | Trees, linear models and small neural networks; Distribution shift; Subgroup error gaps and the model card; Where language models fit, and the capstone | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone: a modelling report and its defence |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Hurbuth Home Care Company** (شركة حُربُث للرعاية المنزلية) is a fictional private provider of home nursing and physiotherapy for older people in an unnamed city; it wants a six-month fall-risk screen built from the nurse's home assessment. Every file is synthetic and written with seed 101 by `labs/prepare_data.py`: 4,776 assessments of 2,500 clients (codes only, staff as roles), a later batch from a newly opened sector, an export with three seeded leaks, and two further screens of the same company for the capstone and the follow-up task. The numbers are not clinical evidence about falls.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | A screening rule by hand | [`D1_1_hand_rule.ipynb`](labs/D1_1_hand_rule.ipynb) |
| 1.2 | Day 1 · Session 2 | Two gradient steps on ten rows | [`D1_2_gradient_steps.ipynb`](labs/D1_2_gradient_steps.ipynb) |
| 1.3 | Day 1 · Session 3 | Two models by hand, then in code | [`D1_3_two_models_by_hand.ipynb`](labs/D1_3_two_models_by_hand.ipynb) |
| 1.4 | Day 1 · Session 4 | Baseline, line and tree on a fixed split | [`D1_4_three_way_split.ipynb`](labs/D1_4_three_way_split.ipynb) |
| 2.1 | Day 2 · Session 1 | Overfit on purpose | [`D2_1_overfit_depth.ipynb`](labs/D2_1_overfit_depth.ipynb) |
| 2.2 | Day 2 · Session 2 | Grouped folds and six curves | [`D2_2_cross_validation.ipynb`](labs/D2_2_cross_validation.ipynb) |
| 2.3 | Day 2 · Session 3 | Four cells by hand, then in code | [`D2_3_confusion_matrix.ipynb`](labs/D2_3_confusion_matrix.ipynb) |
| 2.4 | Day 2 · Session 4 | A threshold from costs | [`D2_4_threshold.ipynb`](labs/D2_4_threshold.ipynb) |
| 3.1 | Day 3 · Session 1 | Visit minutes against two baselines | [`D3_1_regression_baselines.ipynb`](labs/D3_1_regression_baselines.ipynb) |
| 3.2 | Day 3 · Session 2 | A column transformer for the screen | [`D3_2_column_transformer.ipynb`](labs/D3_2_column_transformer.ipynb) |
| 3.3 | Day 3 · Session 3 | Selection before the split | [`D3_3_pipeline_leak.ipynb`](labs/D3_3_pipeline_leak.ipynb) |
| 3.4 | Day 3 · Session 4 | Three seeded leaks | [`D3_4_seeded_leaks.ipynb`](labs/D3_4_seeded_leaks.ipynb) |
| 4.1 | Day 4 · Session 1 | Three families and a baseline | [`D4_1_three_families.ipynb`](labs/D4_1_three_families.ipynb) |
| 4.2 | Day 4 · Session 2 | A later batch from a new sector | [`D4_2_later_batch.ipynb`](labs/D4_2_later_batch.ipynb) |
| 4.3 | Day 4 · Session 3 | Subgroup error rates and a model card | [`D4_3_subgroups_model_card.ipynb`](labs/D4_3_subgroups_model_card.ipynb) |
| 4.4 | Day 4 · Session 4 | The capstone starter | [`D4_4_capstone_start.ipynb`](labs/D4_4_capstone_start.ipynb) |
<!-- /academy:labs -->

## Capstone

A modelling report on a provided public or synthetic dataset, with a baseline, two models, a threshold decision and a model card, defended in ten minutes; the follow-up task repeats it on a second dataset

**Submission:** Individual. A public GitHub repository link holding the modelling report notebook on the admission screen (`labs/capstone/`), run end to end with its fixed seed and its history kept: a baseline, two models, a threshold decision from the stated costs and base rate, and a one-page model card. Then a ten-minute defence: six minutes of talk and four of questions, in which the participant re-runs the notebook and explains a result the instructor selects. The brief, data dictionary, costs and rubric are in [`labs/capstone/README.md`](labs/capstone/README.md); the follow-up task is in [`labs/followup/README.md`](labs/followup/README.md).

| Rubric criterion | Points |
|---|---|
| Held-out workflow: a baseline and two models compared on validation with a fixed seed, the test set opened once, the notebook running end to end | 25 |
| Leakage-safe data handling: the split by date, the field recorded after the outcome removed, every preprocessing step fitted on training rows inside a pipeline | 15 |
| Threshold decision: chosen from the stated error costs and the base rate, with the confusion matrix and metrics at that threshold | 20 |
| Model card: intended use, data, metrics by subgroup with counts, known limits and who should not rely on the model, with an owner and a different reviewer | 20 |
| Defence: re-runs the notebook and explains the result the instructor selects | 20 |

## Assessment

| Component | Weight |
|---|---|
| Supervised check without devices: a hand fit, curve reading and leak spotting | 30% |
| Graded labs: 1.3, 2.1, 2.4, 3.4 and 4.3, 6% each | 30% |
| Capstone with defence | 40% |

- **Passing:** 60% overall and at least 50% on the supervised check
- **Authenticity:** In the defence the participant re-runs the notebook and explains a result the instructor selects; notebook history is inspected
- **Certificate:** completion with assessed work

The supervised check is held on the capstone day in two parts of 60 minutes, on paper. Practice questions for every outcome are in [`assessments/bank.json`](assessments/bank.json).

## AI policy

AI-off for the hand fits, the first attempt at each lab and the supervised check. AI-tutor for debugging in the labs. AI-full for capstone code, which the participant explains in the defence.

Only data classified Public may go into a generative AI tool, never anything classified Restricted or higher, following SDAIA's Generative Artificial Intelligence Guidelines for Government (section 3.5). Every dataset in this course is synthetic and public.

## Responsible use

The subgroup lab measures error gaps across groups in synthetic data and shows how a proxy feature carries a protected attribute; the model card must say who should not rely on the model. Only public or synthetic data is used

## Hours

| Part | Hours | Minutes, as planned |
|---|---|---|
| Instruction | 8 | 480 = 16 sessions × 30 minutes of ideas, checkpoints and debriefs |
| Lab | 10 | 600 = eight labs of 40 minutes and eight of 35 |
| Guided self-study | 5 | 300 = pre-work 75 + after each of Days 1 to 3, 20 + capstone build after Day 4, 105 + follow-up task 60 |
| Assessment | 2 | 120 = the supervised check, two parts of 60, on the capstone day |
| Capstone | 5 | 300 = opening 15 + final preparation 45 + 16 defences of 10 with 2 of changeover each, 192 + trainer review 28 + follow-up brief 20 |
| **Total** | **30** | **1,800** |

Each teaching session is 65 or 70 minutes: 30 of instruction and a lab of 35 or 40.

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

### National references

- [SDAIA: AI Ethics Principles (English)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [سدايا: مبادئ أخلاقيات الذكاء الاصطناعي (العربية)](https://sdaia.gov.sa/ar/SDAIA/about/Documents/ai-principles.pdf)
- [SDAIA: Generative Artificial Intelligence Guidelines for Government (English)](https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf)
- [سدايا: مبادئ الذكاء الاصطناعي التوليدي للجهات الحكومية (العربية)](https://sdaia.gov.sa/ar/SDAIA/about/Files/GenAIGuidelinesForGovernmentARCompressed.pdf)

### Methods

- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Kapoor and Narayanan: Leakage and the Reproducibility Crisis in ML-based Science](https://arxiv.org/abs/2207.07048)
- [Mitchell et al.: Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

### Tools

- [scikit-learn on PyPI](https://pypi.org/project/scikit-learn/)
- [Google Colaboratory: frequently asked questions](https://research.google.com/colaboratory/faq.html)
<!-- /academy:sources -->

## National references

- **SDAIA AI Ethics Principles** (cover dated 2025), Principle 1, Fairness: official text at https://sdaia.gov.sa/ar/SDAIA/about/Documents/ai-principles.pdf (Arabic) and https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf (English). Session 4.3 states only what both editions say; where they differ, this course follows the Arabic. Testing inputs for proxies of a sensitive attribute is named in the English edition and is kept here as the course's own rule.
- **SDAIA Generative Artificial Intelligence Guidelines for Government** (cover dated 2025), section 3.5: the rule on which data may go into generative AI tools, applied in the AI policy above.
