---
course: "How Machine Learning Works"
code: "FND-AIC-101"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-aic-101/"
---

# How Machine Learning Works

Work out by hand and then in code how a model learns from labelled examples, test it honestly on held-out data, set its threshold from error costs, catch leaks and write down its limits

Participants work out how supervised learning turns labelled examples into predictions, by hand first and then in code. They fit a nearest-neighbour rule and a linear model on paper, make a model overfit on purpose, choose a decision threshold from error costs and base rates, and hunt leaks seeded in a pipeline. Labs run on a laptop with small public or synthetic tabular data, and the capstone is an honest modelling report with a model card, defended in person.

- **Audience:** Beginners. Graduates, career changers and computing professionals preparing for Developer or Specialist courses, who can run a Python notebook but have not yet studied AI methods.
- **Prerequisites:** FND-PRG-101 Python Programming for Data and AI; Writes short Python functions and loops; a passed placement exercise is accepted instead of FND-PRG-101; Uses school algebra; the derivative of a squared error is taught in the pre-work
- **Case study:** Hurbuth Home Care Company: Hurbuth (حُربُث) is a fictional private provider of home nursing and physiotherapy for older people in an unnamed city with five sectors; it wants a six-month fall-risk screen that turns the nurse's home assessment into a flag for a physiotherapy home programme. Every file is synthetic and written with a fixed seed by labs/prepare_data.py: 4,776 assessments of 2,500 clients (codes only), a later batch from a newly opened sector, an export with three seeded leaks, and two further screens of the same company for the capstone and the follow-up task.

## Outcomes

- LO1 (Awareness): Explain how a supervised learner fits a prediction rule by minimising a loss, given a ten-row table and a one-feature linear model
- LO2 (Awareness): Distinguish training, validation and test data, and identify overfitting and underfitting from plotted curves, given six complexity and learning curves
- LO3 (Awareness): Identify target, duplicate, preprocessing and temporal leakage in modelling pipelines, given eight short pipeline descriptions
- LO4 (Applied): Implement a train, validate and test workflow comparing a baseline, a linear model and a decision tree, given a tabular dataset and a starter notebook
- LO5 (Applied): Compute confusion-matrix metrics and choose a threshold from error costs and the base rate, given test-set scores and a cost per error type
- LO6 (Applied): Draft a one-page model card with intended use, data, metrics by subgroup and known limits, given the participant's capstone model

## Day 1: Learning from examples

Features and labels, a loss and its gradient, nearest neighbours and a line by hand, and the first honest split

#### [day1#title] Learning from examples

*SDAIA Academy · How Machine Learning Works*

Day 1: Features and labels, a loss and its gradient, nearest neighbours and a line by hand, and the first honest split

Data & AI Foundations Track · 4 teaching days + capstone day · 4 sessions · 4 labs · FND-AIC-101

#### [day1#agenda] Learning from examples

*Today*

- **Session 1 · Features, labels and prediction rules**: A table of examples, the column to predict, and a rule as a function you can score
- **Session 2 · Loss and gradient descent**: A loss scores a rule over all rows; its slope says which way to move the parameters
- **Session 3 · Nearest neighbours and linear models by hand**: Two ways to turn examples into a rule: vote among the closest rows, or fit a line
- **Session 4 · Training, validation and test sets**: Score a model on rows it never saw, choose on validation, open the test set once

#### [day1#case-hurbuth] Hurbuth Home Care wants to know who is likely to fall

*The case for four days*

- A fictional private provider of **home nursing and physiotherapy** for older people, in an unnamed city with five sectors
- A nurse assesses each client at home; the physiotherapy team can run a **home programme** for some of them
- The question: from the assessment, **will this client fall in the next six months?**
- Every record is **synthetic**; clients are codes, staff are roles

- **Who decides what.** The **clinical director** owns the screen and approves its threshold. The **physiotherapy team lead** owns capacity. The **quality and data protection officer** reviews the model card. You are the **data analyst** who builds and tests it.

#### [day1#setup-check] Open a notebook, run the setup check

*Lab environment*

1. **In a free hosted notebook service**: open a lab from the course page; its first cell copies the course and installs the libraries it needs.
2. **On a laptop**: clone the course, then run the commands on the right.
3. Run `00_setup_check.ipynb`. It prints the library versions and the number of assessments.
4. No model is downloaded and no key is needed: every lab runs on a laptop processor in seconds.

```bash
# file: terminal (local option)
git clone https://github.com/SDAIAAcademy/fnd-aic-101
cd fnd-aic-101/labs
uv sync
uv run jupyter lab
```

#### [day1#ai-policy] Your hands first, an assistant after

*How you may use AI assistants here*

- **AI-off.** Every hand fit, the first attempt at each lab, and the supervised check at the end of the course.
- **AI-tutor.** After your first attempt: an assistant may explain an error message or a library call. It does not write the exercise.
- **AI-full.** The capstone code. You explain every line of it in the defence, and the trainer may ask you to re-run any result.

> **Data.** The course data is synthetic and published, so it may go into an assistant. At work, data classified Restricted or higher does not go into generative AI tools.

### Session 1: Features, labels and prediction rules

#### [day1#s1] Features, labels and prediction rules

*Session 1*

- Name the identifier, the features and the label of a table of assessments
- Write a prediction rule as a function of the features
- Count a rule's two kinds of error on rows it was not written from

#### [day1#examples-table] Supervised learning starts from a table of solved examples

*Learning from examples*

A parallel case: the wound-care nurses of the same company want to know which dressing visits will run over 40 minutes.

| Wound area (cm²) | Dressing layers | Diabetic | Ran over 40 min |
| --- | --- | --- | --- |
| 28.5 | 1 | 0 | 0 |
| 7.6 | 4 | 0 | 0 |
| 7.4 | 1 | 1 | 0 |
| 22.1 | 4 | 0 | 1 |
| 25.1 | 3 | 0 | 1 |
| 30.6 | 4 | 1 | 1 |

Six synthetic wound-care visits (labs/_tools/teaching_examples.py).

#### [day1#features-labels] Features go in, the label comes out, the identifier stays out

*Three kinds of column*

- **Features.** What is known when the prediction is made: wound area, layers, diabetic. Written `X`.
- **Label.** What you want to predict, known for past rows only: ran over 40 minutes. Written `y`.
- **Identifier.** Which visit or client a row is about. Needed to join and audit, useless as a predictor.

> **Ask of every column.** Is it known at the moment the prediction is made? If not, it is not a feature, however well it predicts.

#### [day1#rule-is-function] A rule is a function from features to a prediction

*A prediction rule*

```python
# file: a rule you could write by hand
def rule(visit) -> bool:
    return visit["area_cm2"] >= 15
```

- On the six visits it flags **4** and catches all **3** that ran over
- One flag was needless: the 28.5 cm² wound with one layer
- A model is the same thing, written by a fitting procedure from many rows instead of by you from six

#### [day1#two-errors] A screen makes two different mistakes

*Scoring a rule*

- **A miss.** The visit ran over and the rule did not flag it. For a fall screen: a client who falls and got no programme.
- **A needless flag.** The rule flagged a visit that was fine. For a fall screen: a programme for a client who would not have fallen.

> **What goes wrong.** Counting only "right answers" mixes the two, and a rule that flags nobody can look excellent when the outcome is rare.

#### [day1#fresh-rows] Score a rule on rows you did not write it from

*Where to score*

1. **1 · Look** at a few rows and write a rule
2. **2 · Apply** it to many rows you never looked at
3. **3 · Count** misses and needless flags there

> **Takeaway.** A rule always looks best on the rows that inspired it.

#### [day1#tables-in-python] A dataframe holds the table; a library holds the models

*The tools*

```python
# file: reading a table
import pandas as pd
visits = pd.read_csv("visits.csv")
X = visits[["area_cm2", "layers"]]
y = visits["over_40"]
visits.apply(rule, axis=1)   # one answer per row
```

- A **dataframe library** (pandas, Polars) reads and slices tables
- A **machine learning library** (scikit-learn, tidymodels in R) fits and applies models
- The labs give you helpers for loading; you write the parts that matter

#### [day1#check-s1] Which column belongs in the fall-risk rule?

*Checkpoint*

**Question:** An analyst at a home-care company lists four columns for a rule that runs at the end of the nurse's assessment visit. Which one may the rule use?

- A. The client's code
- B. Seconds the client took to walk four metres in the visit
- C. Whether the client called for help in the following month
- D. Whether the client fell in the following six months

#### [day1#lab-1-1] A screening rule by hand

Lab 1.1 · 35 min

Write a fall-risk rule on paper for ten clients, turn it into a function and count its misses and needless flags on 1,000 assessments

[D1_1_hand_rule.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_1_hand_rule.ipynb)

- [ ] Read the ten rows: name the identifier, the features and the label (5 min)
- [ ] Write a rule on paper that flags a client and apply it to the ten rows (8 min)
- [ ] Turn the rule into a Python function (7 min)
- [ ] Count its errors on 1,000 training assessments: missed fallers and needless flags (10 min)
- [ ] Compare with a colleague's rule and with flagging nobody (5 min)
- [ ] ✱ Stretch: Give your rule one number you can tune and find its best value on the training rows

### Session 2: Loss and gradient descent

#### [day1#s2] Loss and gradient descent

*Session 2*

- Compute a mean squared error by hand for a candidate line
- Take a gradient step by hand and say why the loss falls
- Run gradient descent in code and read its loss curve

#### [day1#debrief-1-1] Flagging nobody was right most often and found no faller

*Lab 1.1 debrief*

| Rule, on 1,000 training assessments | Caught | Missed | Needless | Right |
| --- | --- | --- | --- | --- |
| A fall in the past year, or a walk of 8 s or more | 73 | 77 | 153 | 770 |
| A walk of 7 s or more | 62 | 88 | 188 | 724 |
| Flag nobody | 0 | 150 | 0 | 850 |

Lab 1.1 reference run: 150 of the 1,000 clients fell within six months.

#### [day1#loss-idea] A loss turns "how wrong is this rule" into one number

*From writing rules to fitting them*

- Pick a family of rules with a few numbers in it: a line `minutes = w × area + b`
- For one setting of `w` and `b`, compute the error on every example
- Average the squared errors: the **mean squared error**
- **Fitting** means finding the numbers with the lowest loss

> **Same idea, every model.** A line, a logistic regression, a neural network and a language model are all fitted by making a loss small.

#### [day1#loss-worked] Five wound-care visits, two candidate lines

*A worked loss*

| Area (cm²) | Minutes | Error of w 1.0, b 15 | Error of w 1.2, b 15 |
| --- | --- | --- | --- |
| 4 | 20 | -1.0 | -0.2 |
| 8 | 26 | -3.0 | -1.4 |
| 12 | 27 | 0.0 | 2.4 |
| 16 | 33 | -2.0 | 1.2 |
| 20 | 39 | -4.0 | 0.0 |
| Mean squared error |  | 6.00 | 1.84 |

Error is prediction minus actual. Synthetic visits (teaching_examples.py, D1 S2).

#### [day1#loss-bowl] For a line, the loss is a bowl over w and b

*The shape of the loss*

Figure: The tangent at the red point slopes down to the right, so the step goes right.

#### [day1#gradient-step] Step against the slope, by a small rate

*Gradient descent*

```python
# file: one step
e = w * x + b - y               # errors
slope_w = 2 * mean(e * x)
slope_b = 2 * mean(e)
w = w - rate * slope_w
b = b - rate * slope_b
```

- **From w 1.0, b 15 on the five visits.** slope_w = -56.0, slope_b = -4.0. With rate 0.002 the step gives w 1.112, b 15.008 and a loss of 1.93, down from 6.00.
- **The exact bottom.** Least squares gives w 1.125, b 15.5 and a loss of 1.50. More steps keep closing the gap.

#### [day1#learning-rate] The learning rate decides whether the steps arrive

*The one setting*

- **Too small.** Every step is safe and the walk takes forever.
- **About right.** The loss falls fast, then flattens as the slopes flatten near the bottom.
- **Too large.** Each step jumps over the bottom to a higher wall; the loss grows without end.

> **What goes wrong.** A loss curve that rises or turns into "nan" almost always means the rate is too large or a feature has huge values.

#### [day1#loss-curve-reading] Plot the loss after every step and read three things

*Reading a loss curve*

- **Falling**: the steps are working
- **Flat**: you are near the bottom, or the rate is far too small
- **Rising or jagged**: the rate is too large
- Use a log scale on the loss axis: the early drop is huge, the late gains small

#### [day1#check-s2] Which way does the step move the slope?

*Checkpoint*

**Question:** A line predicts dressing minutes from wound area. At the current line the slope of the loss in w is -40 and the rate is 0.01. What does one gradient step do to w?

- A. It lowers w by 0.4
- B. It raises w by 40
- C. It raises w by 0.4
- D. It leaves w where it is

#### [day1#lab-1-2] Two gradient steps on ten rows

Lab 1.2 · 35 min

Fit a line for visit minutes from the number of medicines: a loss by hand, two gradient steps by hand, then a few thousand in code

[D1_2_gradient_steps.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_2_gradient_steps.ipynb)

- [ ] Guess a line for visit minutes and compute its mean squared error by hand (8 min)
- [ ] Write mse(w, b) and check your hand number (5 min)
- [ ] Take two gradient steps by hand from w = 0 and b = 0 (10 min)
- [ ] Run 2,000 steps in code and plot the loss (7 min)
- [ ] Compare with the exact least-squares line (5 min)
- [ ] ✱ Stretch: Raise the learning rate until the loss explodes, and explain why

### Session 3: Nearest neighbours and linear models by hand

#### [day1#s3] Nearest neighbours and linear models by hand

*Session 3*

- Classify a new client by a vote of its nearest neighbours, by hand
- Explain why the scale of a feature changes the neighbours
- Fit a least-squares line from means by hand and reproduce it in code

#### [day1#debrief-1-2] Two steps took the loss from 1,757 to 123; two thousand reached 49

*Lab 1.2 debrief*

| Line for visit minutes | w (min per medicine) | b (min) | Loss |
| --- | --- | --- | --- |
| Start | 0 | 0 | 1,757.1 |
| After step 1 | 5.008 | 0.798 | 195.6 |
| After step 2 | 6.072 | 1.029 | 123.4 |
| After 2,000 steps | 3.752 | 19.263 | 48.98 |
| Exact least squares | 3.752 | 19.267 | 48.98 |

Lab 1.2 reference run on the ten rows, rate 0.01. Slopes at the start: -500.8 and -79.8.

#### [day1#two-families] Keep the examples, or keep a summary of them

*Two ways to use examples*

- **Nearest neighbours.** Keep every row. For a new case, find the most similar rows and let them vote. No fitting; all the work happens when you predict.
- **A linear model.** Fit a few numbers once, then throw the rows away. Prediction is a sum of features times weights.

#### [day1#distance] Similar means close: a distance over the features

*Nearest neighbours*

| Area (cm²) | Layers | Over 40 | Distance to (10, 1) |
| --- | --- | --- | --- |
| 5 | 1 | 0 | 5.00 |
| 6 | 3 | 0 | 4.47 |
| 18 | 2 | 1 | 8.06 |
| 20 | 4 | 1 | 10.44 |
| 9 | 4 | 1 | 3.16 |
| 22 | 1 | 0 | 12.00 |

- Distance = √(area difference² + layers difference²)
- The three nearest (highlighted) vote 0, 0, 1
- Prediction for a 10 cm² wound with one layer: **will not run over**

Six synthetic visits (teaching_examples.py, D1 S3).

#### [day1#units-trap] Change a unit and the neighbours change

*The trap in distances*

- Record the same wounds in **square metres**: 10 cm² becomes 0.001 m²
- Area differences shrink to almost nothing; layers decide every distance
- The three nearest become the rows with one or two layers: rows 1, 6 and 3 in place of 5, 2 and 1
- Nothing about the wounds changed, only a unit

> **What goes wrong.** A distance adds raw differences, so the column with the biggest numbers wins. Scale the columns first (Day 3).

#### [day1#least-squares] For one feature, the best line comes from means

*A line by hand*

**w** = Σ (x − mean x)(y − mean y) ÷ Σ (x − mean x)²

**b** = mean y − w × mean x

The bottom of the bowl, found in one move instead of many steps.

- **The five wound visits.** mean area 12, mean minutes 29. The formula gives w 1.125 and b 15.5: the same line gradient descent was walking towards.

#### [day1#fit-predict] Every scikit-learn model has the same two moves

*The same two models in code*

```python
# file: fit, then predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)    # stores the rows
knn.predict(new_rows)                                  # votes

line = LinearRegression().fit(X[["area_cm2"]], minutes)
line.coef_, line.intercept_                            # w and b
```

#### [day1#check-s3] Why did the neighbours change?

*Checkpoint*

**Question:** A colleague switched the "walk" column of a neighbour model from seconds to milliseconds, and the three nearest clients of a new client changed. What happened?

- A. The walk now has far larger numbers, so it dominates every distance
- B. The clients' walks were measured again
- C. The model's weights were refitted
- D. The colleague changed k

#### [day1#lab-1-3] Two models by hand, then in code

Lab 1.3 · 40 min

Fit two models by hand, then reproduce them in code: a three-nearest-neighbour vote and a least-squares line on the ten-row table

[D1_3_two_models_by_hand.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_3_two_models_by_hand.ipynb)

- [ ] By hand: distances from two new clients to the ten rows and a three-neighbour vote (10 min)
- [ ] Rescale walk seconds and see whether the vote changes (6 min)
- [ ] By hand: the least-squares line for visit minutes from medicines, from means (10 min)
- [ ] Reproduce both models with scikit-learn and compare with your paper (8 min)
- [ ] Write three sentences on how each model turned examples into a rule (6 min)
- [ ] ✱ Stretch: Repeat the vote with k = 1 and k = 5 and say which you trust

### Session 4: Training, validation and test sets

#### [day1#s4] Training, validation and test sets

*Session 4*

- Explain what each of the three sets is for
- Compare models against a baseline that predicts no fall
- Report a held-out result that no choice has touched

#### [day1#debrief-1-3] Tenths of a second flipped both votes; the line matched to the last digit

*Lab 1.3 debrief*

| Three-neighbour vote | Client P | Client Q |
| --- | --- | --- |
| Walk in seconds | fall | no fall |
| Walk in tenths | no fall | fall |

- **Visit minutes from medicines.** By hand and by LinearRegression: w 3.752, b 19.267. The same line Lab 1.2 reached by steps.

Lab 1.3 reference run. P: one recorded fall, 6.8 s; Q: none, 6.0 s.

#### [day1#why-held-out] What matters is the score on clients the model has not seen

*Generalisation*

- A model is built from past rows and used on future clients
- Its score on its own training rows is always flattering
- Hold some rows back, never fit on them, and score there
- The gap between the two scores is the first thing to look at (Day 2)

#### [day1#three-sets] Train on one, choose on another, report on a third

*Three sets, three jobs*

- **Training.** Every model is fitted here. Seen as often as you like.
- **Validation.** Compare models and settings here. Each choice makes its score a little flattering.
- **Test.** Opened once, after every choice. Its score is the one you report.

> **What goes wrong.** Look at the test score, change something, look again: the test set has become a second validation set.

#### [day1#split-by-date] Split by date when the model will meet the future

*How to split*

**Training** the oldest months → **Validation** the next months → **Test** the most recent months

The wound-care visits: 24 months, split at month 14 and month 19 into 1,390, 535 and 475 visits.

#### [day1#baseline-first] Always score the model that does nothing

*The first comparison*

- **0.914** validation accuracy of "no visit runs over 50 minutes"
- **0.940** validation accuracy of a logistic regression

Wound-care visits, label "over 50 minutes" (8% of visits); teaching_examples.py, D1 S4.

#### [day1#ranking-score] AUC: how often a positive case outscores a negative one

*A score for ranking*

- A model gives each row a **score** between 0 and 1
- Pick one visit that ran over and one that did not, at random
- AUC is the chance the first gets the higher score
- 0.5 is a coin; the do-nothing baseline scores exactly 0.5
- The wound-care logistic regression: **0.968**

#### [day1#three-candidates] A baseline, a line and a tree, on the same split

*The lab's comparison*

```python
# file: the shape of Lab 1.4
models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "decision tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
}
# fit each on training rows, score each on validation, choose, then test once
```

#### [day1#check-s4] Which score goes in the report?

*Checkpoint*

**Question:** An analyst trains five models for a wound-care screen, keeps the one with the best validation AUC (0.91), then scores the test set once (0.88). Which number describes how the screen will do next year?

- A. 0.91, the validation AUC
- B. The mean of the two, about 0.90
- C. The training AUC, which is higher still
- D. 0.88, the test AUC

#### [day1#lab-1-4] Baseline, line and tree on a fixed split

Lab 1.4 · 40 min

Compare a baseline, a logistic regression and a tree on the fixed split by date, then score the test set once

[D1_4_three_way_split.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D1_4_three_way_split.ipynb)

- [ ] Load the split and check sizes and base rates (5 min)
- [ ] Score a baseline that predicts no fall (5 min)
- [ ] Fit a logistic regression and a tree (8 min)
- [ ] Compare on validation: accuracy and ranking (10 min)
- [ ] Choose on validation, score test once (7 min)
- [ ] Write the results table (5 min)
- [ ] ✱ Stretch: Split at random instead of by date and compare

#### [day1#wrap-up] Three things to keep

*Day 1 · wrap-up*

- **A model is a rule.** Features in, a prediction out. Fitting picks the rule's numbers by making a loss small.
- **Two kinds of error.** A miss and a needless flag cost different things; counting right answers hides both.
- **Held out, once.** Train, choose on validation, report the test score you opened once, against a baseline.

#### [day1#next] Day 2: Generalisation and measuring

*Next*

About 20 minutes before Day 2: finish any lab you did not finish, then answer the Day 1 practice questions on the course page without an assistant.

## Day 2: Generalisation and measuring

Overfitting on purpose, cross-validation and learning curves, the confusion matrix, and a threshold chosen from costs

#### [day2#title] Generalisation and measuring

*SDAIA Academy · How Machine Learning Works*

Day 2: Overfitting on purpose, cross-validation and learning curves, the confusion matrix, and a threshold chosen from costs

Data & AI Foundations Track · 4 teaching days + capstone day · 4 sessions · 4 labs · FND-AIC-101

#### [day2#agenda] Generalisation and measuring

*Today*

- **Session 1 · Model complexity, overfitting and regularisation**: More capacity fits the training rows better and new rows worse past a point; regularisation pulls it back
- **Session 2 · Cross-validation and learning curves**: Several folds show how much a score moves; a learning curve shows whether more data would help
- **Session 3 · Confusion matrix, precision and recall**: Four counts behind every classification score, and why accuracy hides a lazy model
- **Session 4 · Thresholds, error costs and base rates**: The threshold is a decision: it follows from what each error costs and how common the outcome is

#### [day2#review-day1] What did fitting do yesterday?

*Day 1 review*

**Question:** A colleague says: "Fitting the wound-care line means the computer searched for the line that passes through the most points." What is the accurate version?

- A. It finds the line that passes through every point
- B. It uses the rule the nurse wrote down
- C. It finds the numbers with the lowest mean squared error
- D. It picks the line with the best test score

### Session 1: Model complexity, overfitting and regularisation

#### [day2#s1] Model complexity, overfitting and regularisation

*Session 1*

- Produce training and validation curves over model complexity
- Tell underfitting from overfitting on a plotted curve
- Use a regularisation setting to reduce overfitting

#### [day2#debrief-1-4] Accuracy barely moved; the ranking score did

*Lab 1.4 debrief*

| On validation (1,263 assessments) | Accuracy | AUC |
| --- | --- | --- |
| Baseline: nobody falls | 0.838 | 0.500 |
| Logistic regression | 0.842 | 0.739 |
| Decision tree, depth 4 | 0.834 | 0.682 |

Lab 1.4 reference run. Chosen on validation: the logistic regression; test, opened once: accuracy 0.841, AUC 0.732.

#### [day2#complexity] A flexible model can learn the noise as well as the pattern

*Model complexity*

- A tree's **depth** is how many questions it may ask; each extra level doubles the possible groups
- With enough depth, every training row gets its own leaf: training error reaches zero
- **Overfitting:** training score keeps rising while the validation score falls
- **Underfitting:** both scores are low; the model is too simple to show the pattern

- **In the case.** Two clients with the same walk time and the same falls can have different outcomes. A tree that separates them is memorising chance.

#### [day2#depth-curve] Wound care: deeper trees, then a falling validation score

*A parallel example*

| Tree depth | Training | Validation |
| --- | --- | --- |
| 1 | .941 | .927 |
| 2 | .949 | .935 |
| 4 | .958 | .935 |
| 8 | .983 | .923 |
| 12 | .999 | .923 |
| no limit | 1.000 | .916 |

> **Read the gap.** The gap between the columns grows with depth; choose the depth where validation peaks, not training.

Accuracy on synthetic wound-care visits (predict a visit over 50 minutes), seed 7, from labs/_tools/teaching_examples.py. Depth 3 gives .953 and .935; depth 6 gives .969 and .929.

#### [day2#regularisation] A penalty on large weights keeps a model simpler

*Regularisation*

- Add to the loss a cost for large weights: the model trades fit for simplicity
- In scikit-learn's logistic regression, `C` is the inverse: **small C, strong penalty**
- For trees, the penalty is a limit: `max_depth`, `min_samples_leaf`
- The strength is chosen on validation, like depth

- **Same picture, other dial.** Strong penalty: underfit. No penalty with many features: overfit. The best setting sits between.

#### [day2#check-s1] Which depth do you report?

*Checkpoint*

**Question:** For a wound-care screen, a tree of depth 12 has training accuracy 0.999 and validation 0.923; a tree of depth 3 has 0.953 and 0.935. Which do you choose, and why?

- A. Depth 12, because its training score is higher
- B. Depth 3, because its validation score is higher
- C. Whichever does better on the test set
- D. Depth 12, because its average of the two is higher

#### [day2#lab-2-1] Overfit on purpose

Lab 2.1 · 40 min

Overfit on purpose by raising tree depth and plot the curves, then pull a logistic regression back with regularisation

[D2_1_overfit_depth.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_1_overfit_depth.ipynb)

- [ ] A one-question tree and an unlimited tree: training and validation scores (6 min)
- [ ] Scores for depths 1 to 20 (8 min)
- [ ] Plot the two curves and mark underfitting, the best depth and overfitting (8 min)
- [ ] Regularise a logistic regression from strong to weak (8 min)
- [ ] Choose the depth on validation and score the test set once (5 min)
- [ ] Write the diagnosis in three sentences (5 min)
- [ ] ✱ Stretch: Limit the leaf size in place of the depth and compare the best validation score

### Session 2: Cross-validation and learning curves

#### [day2#s2] Cross-validation and learning curves

*Session 2*

- Run cross-validation with folds grouped by client
- Read the spread of fold scores
- Diagnose underfitting, overfitting and a good fit from complexity and learning curves

#### [day2#debrief-2-1] The unlimited tree was perfect on training and worst on validation

*Lab 2.1 debrief*

| Fall screen, AUC | Training | Validation |
| --- | --- | --- |
| Stump (depth 1) | 0.623 | 0.626 |
| Depth 6, the validation peak | 0.780 | 0.694 |
| No depth limit | 1.000 | 0.563 |

Lab 2.1 reference run. Depth 2 reached 0.691; depth 6 scored 0.668 on test. Regularisation: C = 0.001 gave 0.736 / 0.705, C = 1,000 gave 0.942 / 0.538.

#### [day2#cv-idea] Several folds show how much a score moves by chance

*Cross-validation*

1. **1 · Cut** the training rows into five folds
2. **2 · Rotate** fit on four, score the fifth, five times
3. **3 · Report** the mean and the spread of five scores

> **Repeat clients.** If one client's visits sit in two folds, the model is scored on someone it has seen. Group the folds by client.

#### [day2#cv-example] Wound care: five grouped folds, one spread

*A parallel example*

| Fold | AUC |
| --- | --- |
| 1 | 0.903 |
| 2 | 0.932 |
| 3 | 0.920 |
| 4 | 0.960 |
| 5 | 0.917 |

**0.927 ± 0.019** mean and standard deviation of the five folds

Logistic regression on the synthetic wound-care visits, folds grouped by client.

#### [day2#learning-curves] Training and validation scores as the rows grow

*Learning curves*

- **Good fit.** The two curves meet at a useful score. More rows help a little.
- **Overfit.** Training stays near perfect, validation well below. More rows or a simpler model help.
- **Underfit.** Both curves meet low. More rows do not help; better features or a richer model do.

> **A question to ask.** Will collecting more assessments help? The curve answers before anyone collects them.

#### [day2#check-s2] Is the new model better?

*Checkpoint*

**Question:** On the wound-care screen, five grouped folds give a mean AUC of 0.927 with a spread of 0.019. A new model scores 0.931 on the same folds. What do you conclude?

- A. The new model is better and should replace the old one
- B. The new model is worse, because it is more complex
- C. Regroup the folds by sector and score again
- D. They are tied; prefer the simpler one

#### [day2#lab-2-2] Grouped folds and six curves

Lab 2.2 · 35 min

Cross-validate a tree with folds grouped by client, draw a learning curve, then diagnose six curves and name a remedy for each

[D2_2_cross_validation.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_2_cross_validation.ipynb)

- [ ] Five folds grouped by client on the training period: score a depth-4 tree (8 min)
- [ ] Read the spread across folds against one validation number (5 min)
- [ ] Draw a learning curve from 100 rows to all training rows (8 min)
- [ ] Diagnose six curves and name a remedy for each (14 min)
- [ ] ✱ Stretch: Compare grouped folds with plain folds on the same rows

### Session 3: Confusion matrix, precision and recall

#### [day2#s3] Confusion matrix, precision and recall

*Session 3*

- Fill the four cells of a confusion matrix by hand
- Compute precision, recall and specificity from the cells
- Explain why accuracy misleads at a low base rate

#### [day2#debrief-2-2] Five folds spread from 0.568 to 0.704

*Lab 2.2 debrief*

- Grouped folds on 1,486 training clients (585 with repeat visits): mean AUC **0.657**, spread **0.047**
- The single validation score was 0.682: one draw from a wide range
- The logistic regression's learning curve met near 0.73 to 0.74: a good fit; more rows add little

| Curve | Label |
| --- | --- |
| A, B | overfit, underfit |
| C, D | good fit, underfit |
| E, F | overfit, overfit |

#### [day2#four-cells] Four cells: caught, needless, missed and correctly passed

*The confusion matrix*

|  | Flagged | Not flagged |
| --- | --- | --- |
| **Fell** | caught (true positive, TP) | missed (false negative, FN) |
| **Did not fall** | needless (false positive, FP) | correct pass (true negative, TN) |

> **A threshold makes the flag.** A model gives a score from 0 to 1; flag when the score is at or above the threshold. Change it and the four cells change.

#### [day2#rates] Precision and recall answer two different questions

*Rates from the cells*

- **Recall = TP / (TP + FN).** Of those who fell, the share we flagged. Low recall: fallers missed.
- **Precision = TP / (TP + FP).** Of those we flagged, the share who fell. Low precision: programmes wasted.

- **Specificity** = TN / (TN + FP): of those who did not fall, the share left alone
- **Accuracy** = (TP + TN) / all: dominated by the larger group when one outcome is rare

#### [day2#accuracy-trap] Wound care at 0.5: accuracy 0.940, recall 0.435

*A parallel example*

| 535 visits | Flagged | Not flagged |
| --- | --- | --- |
| Over 50 min | 20 | 26 |
| Not over | 6 | 483 |

- Precision 20 / 26 = **0.769**
- Recall 20 / 46 = **0.435**
- Specificity 483 / 489 = 0.988
- At threshold 0.2: recall 0.783, precision 0.600, accuracy 0.936

#### [day2#check-s3] What does this screen miss?

*Checkpoint*

**Question:** A wound-care screen flags 30 of 500 visits; 24 of the 30 ran long, and 60 visits ran long in all. What is its recall?

- A. 0.40: it caught 24 of the 60 long visits
- B. 0.80: 24 of the 30 flagged visits ran long
- C. 0.06: it flagged 30 of the 500 visits
- D. 0.92: it was right on 458 of the 500 visits

#### [day2#lab-2-3] Four cells by hand, then in code

Lab 2.3 · 35 min

Fill the confusion matrix of a fixed screen by hand on twenty rows, then compute precision, recall and specificity on the validation scores

[D2_3_confusion_matrix.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_3_confusion_matrix.ipynb)

- [ ] Fill the four cells by hand at threshold 0.2 on twenty rows (8 min)
- [ ] Write confusion(y, score, t) (7 min)
- [ ] Precision, recall and specificity on the validation scores (8 min)
- [ ] Accuracy against the base rate (5 min)
- [ ] Check your numbers with scikit-learn (7 min)
- [ ] ✱ Stretch: Repeat at threshold 0.15 and say which cells moved

### Session 4: Thresholds, error costs and base rates

#### [day2#s4] Thresholds, error costs and base rates

*Session 4*

- Compute the expected cost of a threshold from the confusion matrix and two costs
- Choose a threshold on validation scores and report it on test scores
- Explain how a change in the base rate moves the cost of a threshold

#### [day2#debrief-2-3] At 0.5, the fall screen was 84% accurate and missed 188 of 205 fallers

*Lab 2.3 debrief*

| Validation at 0.5 | Flagged | Not |
| --- | --- | --- |
| Fell | 17 | 188 |
| Did not fall | 12 | 1,046 |

- Recall **0.083**, precision 0.586
- Accuracy 0.842, against 0.838 for flagging nobody
- Twenty rows at 0.2: 2 caught, 3 needless, 1 missed, 14 right

#### [day2#costs] Choose the threshold with the lowest expected cost

*Error costs*

- Give each error a cost in the same units: a missed outcome `C_FN`, a needless flag `C_FP`
- Expected cost = C_FN × FN + C_FP × FP, per 1,000 clients
- Try thresholds on **validation**; keep the cheapest; report it once on test
- The owner of the decision sets the costs; the analyst shows the trade-off

> **A rule of thumb.** If scores are well calibrated, flag when the score exceeds C_FP / (C_FP + C_FN).

#### [day2#cost-example] Wound care: a missed long visit costs four short ones

*A parallel example*

| Threshold | Cost per 1,000 | Flagged |
| --- | --- | --- |
| 0.1 | 112 | 0.161 |
| 0.2 | 120 | 0.112 |
| 0.3 | 159 | 0.077 |
| 0.5 | 206 | 0.049 |

> **Rule and search can differ.** The search on validation found 0.1; the rule of thumb, 1 / (1 + 4), says 0.2. The rule assumes calibrated scores.

Costs 4 for a missed long visit and 1 for a needless slot.

#### [day2#base-rate] The same screen gives a different precision where the outcome is more common

*Base rates*

- The **base rate** is the share with the outcome, before any model
- Same scores, more fallers: more of the flagged are fallers, so precision rises
- The cheapest threshold moves too, because misses happen more often

- **What goes wrong.** A precision measured in one sector is quoted for another with twice the base rate. Report the base rate with every precision.

#### [day2#check-s4] What does the rule of thumb give?

*Checkpoint*

**Question:** For a wound-care screen, a missed long visit costs 9 units and a needless extra slot costs 1. With calibrated scores, above which score should a visit be flagged?

- A. 0.5
- B. 9
- C. 0.1
- D. 0.9

#### [day2#lab-2-4] A threshold from costs

Lab 2.4 · 40 min

Set a threshold for a synthetic screening dataset from the cost of a missed fall and of a needless programme, then report its metrics once on the test scores

[D2_4_threshold.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D2_4_threshold.ipynb)

- [ ] Expected cost of a threshold (8 min)
- [ ] Sweep validation thresholds for the cheapest (8 min)
- [ ] Compare with the cost-ratio rule (6 min)
- [ ] Apply it once to the test scores (8 min)
- [ ] Raise the base rate: what moves (5 min)
- [ ] Justify it, and name who approves (5 min)
- [ ] ✱ Stretch: Add a capacity limit of one client in four

#### [day2#wrap-up] Three things to keep

*Day 2 · wrap-up*

- **Choose on validation.** Training scores always favour flexibility. The validation peak is the choice.
- **Scores have a spread.** Grouped folds show how much a score moves; smaller differences are ties.
- **Costs set the threshold.** Four cells, two costs and the base rate decide what to flag; 0.5 is a default, not a decision.

#### [day2#next] Day 3: Data and leakage

*Next*

About 20 minutes before Day 3: finish any lab you did not finish, then the Day 2 practice questions on the course page without an assistant.

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

## Day 4: Model families and limits

Trees, linear models and a small neural network, a later batch from a new sector, error gaps by sex, a model card, and the capstone

#### [day4#title] Model families and limits

*SDAIA Academy · How Machine Learning Works*

Day 4: Trees, linear models and a small neural network, a later batch from a new sector, error gaps by sex, a model card, and the capstone

Data & AI Foundations Track · 4 teaching days + capstone day · 4 sessions · 4 labs · FND-AIC-101

#### [day4#agenda] Model families and limits

*Today*

- **Session 1 · Trees, linear models and small neural networks**: Three families, one pipeline, one fixed seed: compare them honestly against a baseline
- **Session 2 · Distribution shift**: A model meets data that differ from its training rows: new places, new people, a changed measurement
- **Session 3 · Subgroup error gaps and the model card**: Error rates can differ by group at one threshold, a dropped attribute can return through a proxy, and a model card says so
- **Session 4 · Where language models fit, and the capstone**: A language model learns by the same loss minimisation at a far larger scale; the capstone applies the week to a new screen

#### [day4#review-day3] Which split matches use?

*Day 3 review*

**Question:** A wound-care screen will score next year's visits. The analyst has three years of dated visits, many clients with several. Which split gives an honest test score?

- A. Train on the first two years, test on the third
- B. A random 70/30 split of all visits
- C. Train on the last two years, test on the first
- D. A random split of clients, all years mixed

### Session 1: Trees, linear models and small neural networks

#### [day4#s1] Trees, linear models and small neural networks

*Session 1*

- Describe how a tree, a linear model and a small neural network each draw a boundary
- Compare the three families and a baseline in one reproducible notebook
- Explain what a fixed seed does and does not guarantee

#### [day4#debrief-3-4] The analyst's 0.936 became an honest 0.732

*Lab 3.4 debrief*

| Leak | Evidence in the reference run |
| --- | --- |
| Target: `physio_referral` | AUC alone 0.880; updated during the six months after the visit |
| Duplicate: the random split | 573 repeated rows; 236 of 1,605 test rows also in training |
| Temporal: `client_falls_total`, the split | 10.7% of rows count later visits; 30.2% of training rows from 2025 |

Lab 3.4 reference run. The best clinical feature alone, walk_seconds, scored 0.687.

#### [day4#families] Each family draws the boundary between flag and pass differently

*Three families*

- **Linear model.** A weighted sum of the inputs; the boundary is a straight line, or a flat plane. Each weight can be read.
- **Decision tree.** Yes-or-no questions on one input at a time; the boundary is a staircase of boxes.
- **Small neural network.** Several weighted sums, each bent, then summed again; the boundary can curve.

> **All fitted the same way.** The line and the network by gradient steps on a loss, as on Day 1; the tree by choosing one split at a time.

#### [day4#seeds] A fixed seed makes a run repeatable, not certain

*Reproducibility*

- A network starts from random weights; a tree breaks ties at random: the **seed** fixes those draws
- Same seed, same data, same library versions: the same numbers, every time
- Another seed gives a slightly different model: that change is part of the result's uncertainty
- Two models closer than the seed-to-seed change are **tied**

> **What goes wrong.** A notebook that only works when its cells are run in one hidden order. Restart and run all before reporting.

#### [day4#check-s1] What do you report?

*Checkpoint*

**Question:** On a wound-care screen, a small network scores a validation AUC of 0.873 with seed 1 and 0.866 with seed 2; a logistic regression scores 0.869. Which choice is best supported?

- A. The network, quoting its seed 1 score of 0.873
- B. The network, because its mean is 0.0005 higher
- C. The logistic regression, because they are tied
- D. Whichever scores higher on the test set

#### [day4#lab-4-1] Three families and a baseline

Lab 4.1 · 40 min

Train a baseline, a logistic regression, a decision tree and a small neural network through one pipeline with a fixed seed, and report held-out results

[D4_1_three_families.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_1_three_families.ipynb)

- [ ] One preprocessing pipeline for every model (6 min)
- [ ] Fit a baseline, a logistic regression, a tree and a small neural network (10 min)
- [ ] The validation table with a fixed seed (8 min)
- [ ] Run again, then with another seed: what changes (6 min)
- [ ] Choose, score the test set once and save the table (10 min)
- [ ] ✱ Stretch: Change the size of the neural network's hidden layer and watch the validation score

### Session 2: Distribution shift

#### [day4#s2] Distribution shift

*Session 2*

- Compare a later batch with the training rows feature by feature
- Measure a screen's flag rate and recall by sector on a later batch
- Name a shift and the check that would have caught it

#### [day4#debrief-4-1] The network tied with the logistic regression

*Lab 4.1 debrief*

| Validation AUC, seed 101 | AUC |
| --- | --- |
| Baseline | 0.500 |
| Logistic regression | 0.744 |
| Decision tree, depth 4 | 0.685 |
| Small neural network, 16 units | 0.743 |

Seed 7 gave the network 0.743 again. Chosen: the logistic regression; test AUC, opened once: 0.731.

#### [day4#shift-kinds] A model meets rows unlike its training rows

*Distribution shift*

- **New people.** A new sector, older clients, a new referral route: the inputs' mix changes.
- **A changed measurement.** A new device, unit or protocol: the same client gets a different number.
- **A changed outcome rate.** More or fewer events than in training: precision and the cheapest threshold move.

> **The model cannot tell you.** It scores whatever it is given. The checks must be built around it.

#### [day4#monitoring] Cheap checks catch shifts in the first week

*Monitoring*

- **Share flagged**, by sector and by month, against the training period
- **The mean of each input**, by sector and by month, against the training period
- **Recall and precision** once the outcomes arrive, months later
- An owner for each alert, and a person who decides on a pause

> **A parallel example.** A new wound clinic records area in cm², not mm²: every area is 100 times smaller. Its mean area, and its flag rate, would show it on the first day.

#### [day4#check-s2] Which check comes first?

*Checkpoint*

**Question:** A wound-care screen goes live in a new clinic on Sunday. Outcomes take 30 days to arrive. Which check can warn of a problem in the first week?

- A. The clinic's recall at the threshold
- B. The clinic's AUC on its first week of visits
- C. Retraining the model on the clinic's visits
- D. The clinic's share flagged and input means

#### [day4#lab-4-2] A later batch from a new sector

Lab 4.2 · 35 min

Score a later batch that includes a newly opened sector, find what moved, correct it and note it for the model card

[D4_2_later_batch.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_2_later_batch.ipynb)

- [ ] Score the later batch with the chosen screen (5 min)
- [ ] Flag rates and recall by sector (8 min)
- [ ] Compare feature distributions: training rows against the batch (10 min)
- [ ] Find the cause and correct it (7 min)
- [ ] Note what the model card must say (5 min)
- [ ] ✱ Stretch: Compare the base rate of the new sector with the training period

### Session 3: Subgroup error gaps and the model card

#### [day4#s3] Subgroup error gaps and the model card

*Session 3*

- Measure recall and false positive rate by sex, with their counts
- Show how a proxy feature carries a protected attribute
- Draft a one-page model card with intended use, data, metrics by subgroup and limits

#### [day4#debrief-4-2] Central was flagged 100% of the time because its walk was timed over six metres

*Lab 4.2 debrief*

| Later batch, threshold 0.13 | Flagged | Recall |
| --- | --- | --- |
| Central, as recorded | 1.000 | 1.000 |
| Central, walk converted to four metres | 0.517 | 0.773 |
| Other sectors | 0.38–0.46 | 0.68–0.74 |

Central's mean walk 9.56 s against 6.00 in training (ratio 1.59). Central's own AUC 0.724 before and after.

#### [day4#subgroup-rates] One threshold can give different recall to different groups

*Errors by group*

| Wound care, threshold 0.2 | Visits | Long | Recall |
| --- | --- | --- | --- |
| Age 65–74 | 244 | 25 | 0.720 |
| Age 75–84 | 222 | 17 | 0.824 |
| Age 85+ | 69 | 4 | 1.000 |

- Report **recall and false positive rate** per group: they fail people differently
- Report **the counts**: a recall of 1.000 on 4 visits says little

#### [day4#proxy] Leaving an attribute out does not remove it

*Proxies*

- A **proxy** is an input that carries a sensitive attribute: a first name, a home address, a job title
- Test it: predict the attribute from the input; compare with guessing the larger group
- An input can also be **recorded differently** for one group, so the model is weaker for that group even with no proxy

> **What goes wrong.** "The model does not use sex, so it cannot treat men and women differently." Measure, then say.

#### [day4#ethics-principles] The SDAIA AI Ethics Principles: fairness through the model's life

*National reference*

- **Preparing input data:** sensitive attributes are not model inputs unless there is a justification
- **Building and validating:** the chosen model's fairness metrics fall within a defined threshold
- **Deploying and monitoring:** the predefined fairness metrics are monitored in production

> **Course rule.** Also test for proxies of a sensitive attribute. The English edition names this step and the Arabic edition does not, so the course states it as its own rule.

#### [day4#model-card] One page that says what the model is for, and for whom it is weak

*The model card*

- Intended use and out of scope
- Data, split and metrics, against a baseline
- Metrics by group, with counts
- Known limits and **who should not rely on it**
- Monitoring: what, how often, by whom

- **Prepared, reviewed, accepted.** The analyst prepares it; a different person reviews it; the owner of the decision accepts the model for use.

#### [day4#check-s3] What does the model card say?

*Checkpoint*

**Question:** A wound-care screen has recall 0.82 for clients aged 75 to 84 (17 long visits) and 0.72 for 65 to 74 (25 long visits). What belongs in its model card?

- A. The overall recall only, since the groups are small
- B. Both rates with their counts, and a note that the gap is uncertain
- C. A statement that the screen is biased against younger clients
- D. A plan to add age as an input to fix the gap

#### [day4#lab-4-3] Subgroup error rates and a model card

Lab 4.3 · 40 min

Measure subgroup error rates and draft a model card: recall and false positive rate by sex at the chosen threshold, the proxy that carries sex, and who should not rely on the screen

[D4_3_subgroups_model_card.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_3_subgroups_model_card.ipynb)

- [ ] Error rates by sex at the chosen threshold (10 min)
- [ ] The counts behind each rate (6 min)
- [ ] The proxy: predict sex from nurse preference (8 min)
- [ ] Draft the model card from the template (12 min)
- [ ] Write who should not rely on the screen (4 min)
- [ ] ✱ Stretch: Add nurse preference to the screen and compare the gap

### Session 4: Where language models fit, and the capstone

#### [day4#s4] Where language models fit, and the capstone

*Session 4*

- Place a language model on the map of supervised learning
- Read the capstone brief and its rubric
- Start the capstone from a starter notebook that runs end to end

#### [day4#debrief-4-3] The screen caught 80% of female fallers and 59% of male fallers

*Lab 4.3 debrief*

| Test, at 0.13 | Assessments | Fallers | Recall | False pos. rate |
| --- | --- | --- | --- | --- |
| Women | 836 | 146 | 0.801 | 0.430 |
| Men | 600 | 99 | 0.586 | 0.333 |

Synthetic data with a planted pattern: men's past falls are recorded less often. Mean recorded prior falls of those who fell: women 0.93, men 0.26. Nurse preference predicted sex with accuracy 0.913 (the larger group alone: 0.582).

#### [day4#language-models] A language model learns like a supervised model, on text, at a vast scale

*Where language models fit*

- Features: the text so far. Label: the next piece of text, taken from the text itself
- Fitted by the same idea as Day 1: gradient steps that make a loss small
- The family is a very large neural network; the data are huge amounts of text

- **The same limits.** Leakage (test questions seen in training), shift (new topics), and gaps between groups apply here too.

> **Next.** FND-AIC-102 How Language Models Work takes this further.

#### [day4#capstone-brief] A modelling report on a new screen: unplanned admission in 90 days

*The capstone*

- Split by date, a baseline and two models in pipelines
- A leak check, and a threshold from the stated costs and base rate
- A one-page model card with an owner and a different reviewer
- A public repository, its history kept; a ten-minute defence

- **AI and data.** AI-full for the capstone's code; you explain every line. Into any generative AI tool, only public or synthetic data.

#### [day4#capstone-rubric] Five criteria, 100 points

*How the capstone is scored*

| Criterion | Points |
| --- | --- |
| Held-out workflow, end to end with a fixed seed | 25 |
| Leakage-safe data handling | 15 |
| Threshold decision from costs and base rate | 20 |
| Model card with subgroups, limits, owner and reviewer | 20 |
| Defence: re-runs and explains a selected result | 20 |

#### [day4#check-s4] What is a language model's label?

*Checkpoint*

**Question:** Seen as supervised learning, what plays the part of the label when a language model is trained?

- A. The next piece of text after the text so far
- B. A score a person gives each answer
- C. The whole text the model has read so far
- D. A topic tag attached to each document

#### [day4#lab-4-4] The capstone starter

Lab 4.4 · 35 min

Run the capstone starter on the admission screen end to end with its fixed seed, check its data for leaks and write the plan of the report

[D4_4_capstone_start.ipynb](https://sdaiaacademy.github.io/fnd-aic-101/labs/D4_4_capstone_start.ipynb)

- [ ] Read the capstone brief and the data dictionary (5 min)
- [ ] Run the starter end to end with the fixed seed (8 min)
- [ ] Check the data with the leak checklist of Lab 3.4 (7 min)
- [ ] Plan the report: baseline, two models, threshold, model card (10 min)
- [ ] Create the repository and commit the starter (5 min)
- [ ] ✱ Stretch: Work out the cost ratio of the admission screen and the threshold it implies

#### [day4#wrap-up] Three things to keep

*Day 4 · wrap-up*

- **Simple when tied.** Compare families on one pipeline; prefer the simpler one inside the noise.
- **Watch it in use.** Flag rates and input means by group and month catch shifts early.
- **Say who it fails.** Rates by group with counts, proxies tested, and a card that says who should not rely on it.

#### [day4#next] Capstone day: the report and its defence

*Next*

Before capstone day, about 105 minutes: build the report in your repository, commit as you go. On the day: the supervised check without devices, then the defences.

## Capstone: Capstone: a modelling report and its defence

Ten minutes each: six of talk, then four of questions in which the participant re-runs the notebook and explains a result the instructor selects

#### [capstone#title] Capstone: a modelling report and its defence

*SDAIA Academy · How Machine Learning Works*

Capstone: Ten minutes each: six of talk, then four of questions in which the participant re-runs the notebook and explains a result the instructor selects

Data & AI Foundations Track · 4 teaching days + capstone day · 6 min talk and demo · 4 min Q&A · FND-AIC-101

#### [capstone#agenda] How the day runs

*Capstone day*

- **Opening and the supervised check**: the day's plan, then the check on paper, without devices
- **Final preparation**: finish, commit, rehearse once
- **Defences**: talk, then questions with a re-run
- **Review and the follow-up task**: what the defences showed, and the second screen

#### [capstone#supervised-check] Two parts of 60 minutes, on paper, without devices

*The supervised check*

- **A hand fit.** A few rows, a loss, one or two gradient steps or a nearest-neighbour vote.
- **Reading curves.** Training and validation curves: overfit, underfit or a good fit, and what to do next.
- **Spotting leaks.** A data dictionary and a pipeline: which columns and steps leak, and why.

> **To pass the course.** At least 50% on this check, and 60% overall. It counts 30%.

#### [capstone#rubric] The rubric: five criteria, 100 points

*How the capstone is scored*

| Item | Value |
| --- | --- |
| Held-out workflow, fixed seed, test opened once | 25 |
| Leakage-safe data handling | 15 |
| Threshold from costs and base rate | 20 |
| Model card with groups, limits, owner and reviewer | 20 |
| Defence: re-run and explain a selected result | 20 |

#### [capstone#format] Ten minutes: six of talk, four of questions

*The defence*

- **1 · The screen.** What it flags, for whom, and the costs that set its threshold.
- **2 · The evidence.** Baseline, two models, the leak you removed, the test score opened once.
- **3 · The card.** Rates by sex with counts, limits, and who should not rely on it.

> **In the questions.** Restart the notebook, run all, and explain the result the instructor points to.

#### [capstone#running-order] Who defends when

*Running order*

| # | Participant | # | Participant |
| --- | --- | --- | --- |
| 1 |  | 9 |  |
| 2 |  | 10 |  |
| 3 |  | 11 |  |
| 4 |  | 12 |  |
| 5 |  | 13 |  |
| 6 |  | 14 |  |
| 7 |  | 15 |  |
| 8 |  | 16 |  |

#### [capstone#final-prep] Final preparation · 45 minutes

*Before the first defence*

Restart and run all, commit, push, and rehearse six minutes once.

#### [capstone#now-presenting] Talk

*Now defending*

##### Questions and re-run

#### [capstone#review] What the defences showed

*Trainer review*

- **Strongest patterns.** What several reports got right.
- **Common gaps.** What most reports would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#follow-up] The same workflow on a second screen: pressure injury in 90 days

*The follow-up task*

- About 60 minutes, on your own, within two weeks; tracked, not graded
- The brief and data dictionary: `labs/followup/README.md`
- A second notebook in your capstone repository; send the trainer the commit link
- One column is recorded after the outcome: find it before you trust a score

#### [capstone#what-next] Keep going

*After the course*

- **Your repository stays yours:** keep it running from a fresh start
- **FND-AIC-102 How Language Models Work:** the same loss idea at the scale of text
- **DS-204 Machine Learning for Tabular Data:** boosted trees, tuning and calibration

#### [capstone#thanks] Thank you

*SDAIA Academy*

How Machine Learning Works
