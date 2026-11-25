---
course: "How Machine Learning Works"
code: "FND-AIC-101"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-aic-101/slides/day2.html"
---

# How Machine Learning Works

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
