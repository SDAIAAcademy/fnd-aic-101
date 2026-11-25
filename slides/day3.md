---
course: "How Machine Learning Works"
code: "FND-AIC-101"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-aic-101/slides/day3.html"
---

# How Machine Learning Works

## Day 3: Data and leakage

Regression against naive baselines, encoding, scaling and missing values, pipelines fitted on training rows, and four kinds of leak

#### [day3#title] Data and leakage

*SDAIA Academy · How Machine Learning Works*

Day 3: Regression against naive baselines, encoding, scaling and missing values, pipelines fitted on training rows, and four kinds of leak

Data & AI Foundations Track · 4 teaching days + capstone day · 4 sessions · 4 labs · FND-AIC-101

#### [day3#agenda] Data and leakage

*Today*

- **Session 1 · Regression errors against a naive baseline**: A regression error means little until it beats predicting the average
- **Session 2 · Encoding, scaling and missing values**: Models read numbers: categories need columns, distances need scales, and a gap needs a value and a flag
- **Session 3 · Pipelines fitted on training data only**: Every step that learns from data learns from training rows only, inside each fold
- **Session 4 · Target, duplicate and temporal leakage**: A score that is too good usually means the answer is in the inputs, the test rows are in training, or the future is in the past

#### [day3#review-day2] Where should the threshold come from?

*Day 2 review*

**Question:** A wound-care team asks the analyst to "just use 0.5, it is the standard". What is the best reply?

- A. Agree: 0.5 is where a probability turns into a yes
- B. Ask what a missed long visit and a needless slot each cost
- C. Choose the threshold with the highest accuracy
- D. Choose the threshold that does best on the test set

### Session 1: Regression errors against a naive baseline

#### [day3#s1] Regression errors against a naive baseline

*Session 1*

- Compute mean absolute error and root mean squared error
- Compare a model with the mean and median baselines
- Find where a model errs most from its residuals

#### [day3#debrief-2-4] The cheapest threshold was 0.17, far from 0.5

*Lab 2.4 debrief*

| Per 1,000 validation clients, costs 6 and 1 | Cost |
| --- | --- |
| Flag nobody | 974 |
| Threshold 0.5 | 903 |
| Flag everyone | 838 |
| The rule of thumb, 1/7 = 0.143 | 571 |
| Cheapest on validation, 0.17 | 544 |

On test at 0.17: 32% flagged, recall 0.608, precision 0.322. With the base rate raised to 0.291, the cheapest threshold fell to 0.07.

#### [day3#regression-errors] For a number, measure how far off, on average

*Regression*

- **MAE**, mean absolute error: the typical miss, in the target's units
- **RMSE**, root mean squared error: like MAE, but large misses weigh more
- RMSE is never below MAE; a big gap means a few large misses
- Always compare with a **naive baseline**: predict the training mean, or the median

- **Five errors, by hand.** Errors 3, −2, 10, −1, 0 minutes. MAE = 16 / 5 = **3.20**. RMSE = √(114 / 5) = **4.77**. The 10 pulls RMSE up.

#### [day3#baseline-regression] Wound care: a linear model halves the baseline's error

*A parallel example*

| Predict dressing visit minutes, on validation | MAE |
| --- | --- |
| The training mean, for everyone | 8.30 |
| The training median, for everyone | 8.24 |
| Linear regression on wound and client features | 4.60 |

Synthetic wound-care visits, seed 7. The linear model's RMSE is 5.89.

> **What goes wrong.** A model is reported with an MAE of 4.6 and no baseline: nobody can tell if that is good.

#### [day3#check-s1] Which error is larger?

*Checkpoint*

**Question:** Four wound-care predictions miss by 2, −2, 2 and −10 minutes. Which is true of their MAE and RMSE?

- A. MAE is −2 and RMSE is about 5.3
- B. MAE is 4 and RMSE is 3
- C. MAE is 4 and RMSE is 112
- D. MAE is 4 and RMSE is about 5.3

#### [day3#lab-3-1] Visit minutes against two baselines

Lab 3.1 · 35 min

Predict the length of a nurse visit and compare a linear model with the mean and median baselines on validation rows

[D3_1_regression_baselines.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_1_regression_baselines.ipynb)

- [ ] Two naive baselines: the training mean and median (5 min)
- [ ] Mean absolute error and root mean squared error by hand on five rows (7 min)
- [ ] A linear model on five features (8 min)
- [ ] Errors on validation against the baselines (8 min)
- [ ] Where the model errs most: residuals by group (7 min)
- [ ] ✱ Stretch: Try a decision tree regressor and compare its validation error

### Session 2: Encoding, scaling and missing values

#### [day3#s2] Encoding, scaling and missing values

*Session 2*

- Encode a categorical column as indicator columns
- Impute a missing value from training rows and keep a missing flag
- Explain when scaling changes a model's answer

#### [day3#debrief-3-1] The linear model cut the typical miss from 8.2 to 4.4 minutes

*Lab 3.1 debrief*

| Visit minutes, validation | MAE | RMSE |
| --- | --- | --- |
| Training mean (33.12) | 8.20 | 10.17 |
| Training median (32) | 8.23 | 10.27 |
| Linear regression | 4.41 | 5.48 |

By walking aid: without 810 rows, MAE 4.37, mean error −0.19; with 453 rows, MAE 4.46, mean error +0.53.

#### [day3#column-kinds] Each kind of column needs its own step

*Preparing columns*

| Column | Problem | Step |
| --- | --- | --- |
| sector (four names so far) | models need numbers | one-hot: one 0/1 column per sector |
| walk_seconds, age | different units and ranges | scale: subtract the mean, divide by the spread |
| bmi, sometimes blank | missing values | impute the median, add a "was missing" flag |

> **Learned from training rows.** The mean, the spread, the median and the list of sectors are all numbers a step learns. Learn them on training rows only.

#### [day3#scaling-matters] Wound care: scaling changed a neighbour model's AUC from 0.821 to 0.875

*A parallel example*

- Five nearest neighbours, wound area in **mm²** beside small counts
- Unscaled, the area's large numbers decide every distance
- Scaled, each feature has a say

- **0.821** AUC, unscaled
- **0.875** AUC, scaled

#### [day3#column-transformer] A column transformer applies each step to its columns

*In code*

```python
# file: one preprocessing object
prep = ColumnTransformer([
    ("numbers", StandardScaler(), FEATURES),
    ("bmi", make_pipeline(SimpleImputer(strategy="median",
                                        add_indicator=True),
                          StandardScaler()), ["bmi"]),
    ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
])
model = make_pipeline(prep, LogisticRegression())
```

#### [day3#check-s2] Where does the median come from?

*Checkpoint*

**Question:** Wound area is blank for some visits. The analyst fills blanks with the median. Which median should fill the blanks in the validation rows?

- A. The median of the training rows
- B. The median of the validation rows
- C. The median of all rows together
- D. Zero, because a blank means nothing

#### [day3#lab-3-2] A column transformer for the screen

Lab 3.2 · 35 min

Encode the sector, impute body mass index with a missing flag and scale the numbers in one column transformer, and see scaling change a neighbour model

[D3_2_column_transformer.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_2_column_transformer.ipynb)

- [ ] Find the categorical, numeric and incomplete columns (5 min)
- [ ] One-hot encode the sector (6 min)
- [ ] Impute body mass index from training rows and add a missing flag (8 min)
- [ ] Scale the numbers and compare a neighbour model with and without (10 min)
- [ ] Put it all in one column transformer inside a pipeline (6 min)
- [ ] ✱ Stretch: Compare the missing rate of body mass index for clients who live alone and with others

### Session 3: Pipelines fitted on training data only

#### [day3#s3] Pipelines fitted on training data only

*Session 3*

- Explain why a step fitted on all rows leaks
- Measure the effect of selecting features before the split
- Build a pipeline that refits every step inside each fold

#### [day3#debrief-3-2] Scaling helped the neighbour model; the full pipeline scored 0.744

*Lab 3.2 debrief*

- BMI was blank in 6.1% of training rows; 83 validation rows were filled with the training median, 27.7
- Fifty nearest neighbours: AUC **0.647** unscaled, **0.683** scaled
- The column transformer and a logistic regression: 15 inputs, validation AUC **0.744**

#### [day3#fit-on-train] Fit every step on training rows, inside the pipeline

*The rule*

1. **1 · Split first** before any step learns anything
2. **2 · Fit the pipeline** scaling, filling, selection, model: on training rows
3. **3 · Apply** the fitted pipeline to validation and test

> **In cross-validation too.** The whole pipeline goes inside the folds, so each fold prepares from its own training part.

#### [day3#selection-leak] Noise columns, selected on all rows, looked useful

*A parallel example*

- 150 wound-care visits, **600 columns of pure noise**, keep the 10 best
- Selection on all rows, then cross-validation: AUC **0.783**
- Selection inside the pipeline: AUC **0.439**, no better than chance

- **Why it happens.** With 600 columns, some match the labels by luck. Choosing them with every label in view, then scoring on the same labels, rewards the luck.

#### [day3#check-s3] Which number is honest?

*Checkpoint*

**Question:** A wound-care analyst keeps the 20 best of 900 sensor columns, chosen on all 200 visits, then reports a cross-validated AUC of 0.81. What should a reviewer ask for?

- A. The same procedure with ten folds in place of five
- B. A random forest in place of the logistic regression
- C. The selection inside the pipeline, refitted in every fold
- D. The 5 best columns in place of the 20 best

#### [day3#lab-3-3] Selection before the split

Lab 3.3 · 40 min

Select features on all rows of a pilot whose columns carry no signal, see the score it invents, and remove it with a pipeline

[D3_3_pipeline_leak.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_3_pipeline_leak.ipynb)

- [ ] Meet the pilot: 240 clients and 400 activity columns (4 min)
- [ ] Select the 20 best columns on all rows, then cross-validate (8 min)
- [ ] Do the same with the selection inside a pipeline (8 min)
- [ ] Shuffle the labels and repeat both (8 min)
- [ ] Scaler and imputer fitted on everything: measure the difference (7 min)
- [ ] Write the rule you will follow (5 min)
- [ ] ✱ Stretch: Repeat with 5 and with 100 selected columns

### Session 4: Target, duplicate and temporal leakage

#### [day3#s4] Target, duplicate and temporal leakage

*Session 4*

- Recognise a field recorded after the outcome
- Find copies of rows on both sides of a split
- Spot information from after the prediction date

#### [day3#debrief-3-3] Selection before the split turned noise into an AUC of 0.838

*Lab 3.3 debrief*

| Sensor pilot, 240 clients, 400 noise columns | Before the split | In the pipeline |
| --- | --- | --- |
| Real labels | 0.838 | 0.457 |
| Shuffled labels | 0.904 | 0.533 |

Scaling fitted on all rows against training rows: 0.7403 and 0.7403. Scaling leaked nothing measurable here; selection leaked a great deal.

#### [day3#leak-types] Three leaks a pipeline does not catch

*Leakage*

- **Target leak.** A column recorded **after** the outcome, often because of it: a treatment started because the event happened.
- **Duplicate leak.** The same row on both sides of the split; the model is scored on rows it has seen.
- **Temporal leak.** A column, or the split, that uses what happened after the prediction date.

> **Not rare.** Kapoor and Narayanan found leakage in 329 published papers across 17 fields.

#### [day3#leak-checklist] Four questions before trusting a score

*A checklist*

- **When is each column recorded?** Read the data dictionary; anything after the prediction date goes
- **Does one column alone rank almost perfectly?** Score each alone; a very good one deserves the first question
- **Are there copies?** Count repeated identifiers, and test rows with a twin in training
- **Does the split match use?** The screen predicts next year, so train on the past and test on the latest period

#### [day3#check-s4] Which column is the leak?

*Checkpoint*

**Question:** A screen predicts whether a wound will be infected within 30 days of a visit. Alone, one column ranks infected wounds with an AUC of 0.93. Which column is most likely a leak?

- A. Wound area, measured at the visit
- B. An antibiotic prescription, updated over the 30 days
- C. The client's diabetes status, at registration
- D. The number of past wounds, counted at the visit

#### [day3#lab-3-4] Three seeded leaks

Lab 3.4 · 40 min

Find three seeded leaks in a working pipeline, fix each one and rescore, and write a leak log

[D3_4_seeded_leaks.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D3_4_seeded_leaks.ipynb)

- [ ] Run the working pipeline and note its score (5 min)
- [ ] Look for a field recorded after the outcome (8 min)
- [ ] Look for copies across the split (8 min)
- [ ] Look for information from the future (8 min)
- [ ] Fix all three and rescore (7 min)
- [ ] Write the leak log: each leak, its evidence, its fix (4 min)
- [ ] ✱ Stretch: Fix the leaks one at a time and rank them by how much each moved the score

#### [day3#wrap-up] Three things to keep

*Day 3 · wrap-up*

- **Baseline first.** An error means nothing until a naive baseline is beside it.
- **One pipeline.** Every learned step is fitted on training rows, inside the folds.
- **Known on the day.** Each column must exist when the prediction is made; the split must match use.

#### [day3#next] Day 4: Model families and limits

*Next*

About 20 minutes before Day 4: finish any lab you did not finish, then the Day 3 practice questions on the course page without an assistant.
