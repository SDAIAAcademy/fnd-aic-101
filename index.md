---
course: "How Machine Learning Works"
code: "FND-AIC-101"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-aic-101/index.html"
---

# How Machine Learning Works

كيف يعمل تعلم الآلة

Data & AI Foundations Track · 4 teaching days + capstone day · 30 hours · Level 100

Work out by hand and then in code how a model learns from labelled examples, test it honestly on held-out data, set its threshold from error costs, catch leaks and write down its limits

Participants work out how supervised learning turns labelled examples into predictions, by hand first and then in code. They fit a nearest-neighbour rule and a linear model on paper, make a model overfit on purpose, choose a decision threshold from error costs and base rates, and hunt leaks seeded in a pipeline. Labs run on a laptop with small public or synthetic tabular data, and the capstone is an honest modelling report with a model card, defended in person.

**Who it is for:** Beginners. Graduates, career changers and computing professionals preparing for Developer or Specialist courses, who can run a Python notebook but have not yet studied AI methods.

**Format:** Intensive: 4 teaching days and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Hurbuth Home Care Company.** Hurbuth (حُربُث) is a fictional private provider of home nursing and physiotherapy for older people in an unnamed city with five sectors; it wants a six-month fall-risk screen that turns the nurse's home assessment into a flag for a physiotherapy home programme. Every file is synthetic and written with a fixed seed by labs/prepare_data.py: 4,776 assessments of 2,500 clients (codes only), a later batch from a newly opened sector, an export with three seeded leaks, and two further screens of the same company for the capstone and the follow-up task.

## The course

### Day 1: Learning from examples

Slides: https://sdaiaacademy.github.io/fnd-aic-101/slides/day1.md

- **Session 1: Features, labels and prediction rules**
  - Name the identifier, the features and the label of a table of assessments
  - Write a prediction rule as a function of the features
  - Count a rule's two kinds of error on rows it was not written from
  - [Lab 1.1: A screening rule by hand (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_1_hand_rule.ipynb)
- **Session 2: Loss and gradient descent**
  - Compute a mean squared error by hand for a candidate line
  - Take a gradient step by hand and say why the loss falls
  - Run gradient descent in code and read its loss curve
  - [Lab 1.2: Two gradient steps on ten rows (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_2_gradient_steps.ipynb)
- **Session 3: Nearest neighbours and linear models by hand**
  - Classify a new client by a vote of its nearest neighbours, by hand
  - Explain why the scale of a feature changes the neighbours
  - Fit a least-squares line from means by hand and reproduce it in code
  - [Lab 1.3: Two models by hand, then in code (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_3_two_models_by_hand.ipynb)
- **Session 4: Training, validation and test sets**
  - Explain what each of the three sets is for
  - Compare models against a baseline that predicts no fall
  - Report a held-out result that no choice has touched
  - [Lab 1.4: Baseline, line and tree on a fixed split (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_4_three_way_split.ipynb)

### Day 2: Generalisation and measuring

Slides: https://sdaiaacademy.github.io/fnd-aic-101/slides/day2.md

- **Session 1: Model complexity, overfitting and regularisation**
  - Produce training and validation curves over model complexity
  - Tell underfitting from overfitting on a plotted curve
  - Use a regularisation setting to reduce overfitting
  - [Lab 2.1: Overfit on purpose (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_1_overfit_depth.ipynb)
- **Session 2: Cross-validation and learning curves**
  - Run cross-validation with folds grouped by client
  - Read the spread of fold scores
  - Diagnose underfitting, overfitting and a good fit from complexity and learning curves
  - [Lab 2.2: Grouped folds and six curves (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_2_cross_validation.ipynb)
- **Session 3: Confusion matrix, precision and recall**
  - Fill the four cells of a confusion matrix by hand
  - Compute precision, recall and specificity from the cells
  - Explain why accuracy misleads at a low base rate
  - [Lab 2.3: Four cells by hand, then in code (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_3_confusion_matrix.ipynb)
- **Session 4: Thresholds, error costs and base rates**
  - Compute the expected cost of a threshold from the confusion matrix and two costs
  - Choose a threshold on validation scores and report it on test scores
  - Explain how a change in the base rate moves the cost of a threshold
  - [Lab 2.4: A threshold from costs (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_4_threshold.ipynb)

### Day 3: Data and leakage

Slides: https://sdaiaacademy.github.io/fnd-aic-101/slides/day3.md

- **Session 1: Regression errors against a naive baseline**
  - Compute mean absolute error and root mean squared error
  - Compare a model with the mean and median baselines
  - Find where a model errs most from its residuals
  - [Lab 3.1: Visit minutes against two baselines (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_1_regression_baselines.ipynb)
- **Session 2: Encoding, scaling and missing values**
  - Encode a categorical column as indicator columns
  - Impute a missing value from training rows and keep a missing flag
  - Explain when scaling changes a model's answer
  - [Lab 3.2: A column transformer for the screen (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_2_column_transformer.ipynb)
- **Session 3: Pipelines fitted on training data only**
  - Explain why a step fitted on all rows leaks
  - Measure the effect of selecting features before the split
  - Build a pipeline that refits every step inside each fold
  - [Lab 3.3: Selection before the split (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_3_pipeline_leak.ipynb)
- **Session 4: Target, duplicate and temporal leakage**
  - Recognise a field recorded after the outcome
  - Find copies of rows on both sides of a split
  - Spot information from after the prediction date
  - [Lab 3.4: Three seeded leaks (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_4_seeded_leaks.ipynb)

### Day 4: Model families and limits

Slides: https://sdaiaacademy.github.io/fnd-aic-101/slides/day4.md

- **Session 1: Trees, linear models and small neural networks**
  - Describe how a tree, a linear model and a small neural network each draw a boundary
  - Compare the three families and a baseline in one reproducible notebook
  - Explain what a fixed seed does and does not guarantee
  - [Lab 4.1: Three families and a baseline (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_1_three_families.ipynb)
- **Session 2: Distribution shift**
  - Compare a later batch with the training rows feature by feature
  - Measure a screen's flag rate and recall by sector on a later batch
  - Name a shift and the check that would have caught it
  - [Lab 4.2: A later batch from a new sector (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_2_later_batch.ipynb)
- **Session 3: Subgroup error gaps and the model card**
  - Measure recall and false positive rate by sex, with their counts
  - Show how a proxy feature carries a protected attribute
  - Draft a one-page model card with intended use, data, metrics by subgroup and limits
  - [Lab 4.3: Subgroup error rates and a model card (40 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_3_subgroups_model_card.ipynb)
- **Session 4: Where language models fit, and the capstone**
  - Place a language model on the map of supervised learning
  - Read the capstone brief and its rubric
  - Start the capstone from a starter notebook that runs end to end
  - [Lab 4.4: The capstone starter (35 min)](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_4_capstone_start.ipynb)

### Capstone: Capstone: a modelling report and its defence

Slides: https://sdaiaacademy.github.io/fnd-aic-101/slides/capstone.md

## Capstone

A modelling report on a provided public or synthetic dataset, with a baseline, two models, a threshold decision and a model card, defended in ten minutes; the follow-up task repeats it on a second dataset

**What you submit:** Individual. A public GitHub repository link holding the modelling report notebook on the admission screen (labs/capstone/), run end to end with its fixed seed and its history kept: a baseline, two models, a threshold decision from the stated costs and base rate, and a one-page model card. Then a ten-minute defence: six minutes of talk and four of questions, in which the participant re-runs the notebook and explains a result the instructor selects.

**Rubric:**

- Held-out workflow: a baseline and two models compared on validation with a fixed seed, the test set opened once, the notebook running end to end: 25%
- Leakage-safe data handling: the split by date, the field recorded after the outcome removed, every preprocessing step fitted on training rows inside a pipeline: 15%
- Threshold decision: chosen from the stated error costs and the base rate, with the confusion matrix and metrics at that threshold: 20%
- Model card: intended use, data, metrics by subgroup with counts, known limits and who should not rely on the model, with an owner and a different reviewer: 20%
- Defence: re-runs the notebook and explains the result the instructor selects: 20%

## Outcomes

- **LO1** Explain how a supervised learner fits a prediction rule by minimising a loss, given a ten-row table and a one-feature linear model (Awareness) Evidence: a hand-worked loss and two gradient steps, with a three-sentence explanation
- **LO2** Distinguish training, validation and test data, and identify overfitting and underfitting from plotted curves, given six complexity and learning curves (Awareness) Evidence: a diagnosis and remedy per curve, at least five of six correct
- **LO3** Identify target, duplicate, preprocessing and temporal leakage in modelling pipelines, given eight short pipeline descriptions (Awareness) Evidence: each leak named with its fix, at least six of eight correct
- **LO4** Implement a train, validate and test workflow comparing a baseline, a linear model and a decision tree, given a tabular dataset and a starter notebook (Applied) Evidence: a notebook that runs end to end with a fixed seed and reports held-out metrics
- **LO5** Compute confusion-matrix metrics and choose a threshold from error costs and the base rate, given test-set scores and a cost per error type (Applied) Evidence: a metrics table and a justified threshold
- **LO6** Draft a one-page model card with intended use, data, metrics by subgroup and known limits, given the participant's capstone model (Applied) Evidence: a model card reviewed in the defence

## Tools

- **Notebook environment:** JupyterLab, a free hosted notebook service
- **Machine learning library:** scikit-learn, tidymodels in R
- **Dataframes and plotting:** pandas with Matplotlib, Polars with seaborn

## Prerequisites

- FND-PRG-101 Python Programming for Data and AI
- Writes short Python functions and loops; a passed placement exercise is accepted instead of FND-PRG-101
- Uses school algebra; the derivative of a squared error is taught in the pre-work

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

---

SDAIA Academy · Data & AI Foundations Track · How Machine Learning Works

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-25 · commit 193791f
