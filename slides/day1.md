---
course: "How Machine Learning Works"
code: "FND-AIC-101"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-aic-101/slides/day1.html"
---

# How Machine Learning Works

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
