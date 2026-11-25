# Answer key for trainers

The numbers a correct solution run prints (seed 101, the data in this repository), and what each lab plants.
Every number here comes from the solution notebooks in this folder; rerun them after `prepare_data.py` changes.
Do not show this file to participants before the labs.

## Day 1

- **1.1** Sample of 1,000 assessments, 150 fallers. The reference rule (a fall in the past year, or a walk of 8
  seconds or more): 73 fallers caught, 77 missed, 153 needless flags, 770 right. The colleague's rule (walk of 7
  seconds or more): 62 caught, 88 missed, 188 needless. Flagging nobody: 150 missed, 850 right.
- **1.2** Zero line: loss 1,757.1, slopes −500.8 (w) and −79.8 (b). After one step (rate 0.01): w 5.008, b 0.798,
  loss 195.593; after two: 6.072, 1.029, 123.377. After 2,000 steps: w 3.752, b 19.263, loss 48.981; the exact
  least-squares line w 3.752, b 19.267.
- **1.3 (graded)** New clients P (1 fall, 6.8 s) and Q (0 falls, 6.0 s). Three nearest neighbours vote: walk in
  seconds P 1, Q 0; walk in tenths of a second P 0, Q 1 (the units trap). The line on visit minutes: w 3.752,
  b 19.267, through the means (5.5, 39.9).
- **1.4** Training 2,077 (base 0.159), validation 1,263 (0.162), test 1,436 (0.171). Validation: baseline
  accuracy 0.838, AUC 0.500; logistic regression 0.842, 0.739; tree of depth 4 0.834, 0.682. Chosen: logistic
  regression; test baseline accuracy 0.829, logistic regression 0.841 and AUC 0.732. Planted: accuracy barely
  separates the baseline from the models.

## Day 2

- **2.1 (graded)** Stump: training AUC 0.623, validation 0.626. Unlimited tree: 1.000 and 0.563. Validation peak
  at depth 6 (0.694; depth 2 gives 0.691); test AUC at depth 6, 0.668. Regularisation of a degree-3 logistic
  regression on 600 rows: C 0.001 gives 0.736 and 0.705; C 1,000 gives 0.942 and 0.538.
- **2.2** 1,486 clients in training, 585 with more than one assessment. Grouped five-fold AUCs 0.685, 0.668,
  0.704, 0.660, 0.568: mean 0.657, spread 0.047; the single validation AUC was 0.682. The six curves: A unlimited
  tree, overfit; B stump, underfit; C logistic regression, good fit; D logistic regression on two weak features,
  underfit; E depth 18, overfit; F one nearest neighbour, overfit.
- **2.3** Twenty rows at threshold 0.2: 2 caught, 3 needless, 1 missed, 14 right. All of validation at 0.5: 17
  caught, 12 needless, 188 missed, 1,046 right; recall 0.083, precision 0.586, specificity 0.989, accuracy 0.842
  against a base rate of 0.162. Planted: high accuracy with a recall near zero.
- **2.4 (graded)** Costs: a missed fall 6, a needless programme 1. Per 1,000 validation clients: at 0.5, 903;
  flag nobody 974; flag everyone 838. Cheapest on validation 0.17 (544); the cost-ratio rule 1/7 = 0.143 (571).
  Test at 0.17: 149 caught, 314 needless, 96 missed, 877 right; flagged 0.322, recall 0.608, precision 0.322,
  specificity 0.736, cost 619.8. With every test faller counted twice (base 0.291): cheapest 0.07; precision at
  0.17 rises from 0.322 to 0.487.

## Day 3

- **3.1** Training mean visit 33.12 minutes, median 32. Five rows by hand: MAE 10.98, RMSE 12.78. Validation MAE
  and RMSE: mean 8.20 and 10.17, median 8.23 and 10.27, linear model 4.41 and 5.48.
- **3.2** BMI missing in 6.1% of training rows; 83 validation rows filled with 27.7. Fifty nearest neighbours:
  raw AUC 0.647, scaled 0.683. The full column transformer and logistic regression: 15 inputs, validation AUC
  0.744.
- **3.3** Sensor pilot, 240 clients (30 fallers) and 400 noise columns. Selection on all rows before
  cross-validation: AUC 0.838; selection inside the pipeline: 0.457. Shuffled labels: 0.904 and 0.533. Scaling
  fitted on all rows against training rows only: 0.7403 and 0.7403 (scaling leaks little; selection a lot).
- **3.4 (graded)** Planted: `physio_referral` (target leak), 12% duplicated rows with a random split (duplicate
  leak), `client_falls_total` and the random split (temporal leak). The analyst's test AUC 0.936. Alone:
  `physio_referral` 0.880, `client_falls_total` 0.738, `walk_seconds` 0.687. Duplicate rows 573; test rows with a
  copy in training 236 of 1,605. Rows whose total counts later assessments 10.7%; training rows from 2025 30.2%.
  Fixed: test AUC 0.732.

## Day 4

- **4.1** Validation AUC: baseline 0.500, logistic regression 0.744, tree 0.685, small neural network 0.743 (seed
  7: 0.743). Chosen: logistic regression (the network is within 0.005); test AUC 0.731.
- **4.2** Threshold 0.13 (the standard screen, costs 6 to 1). The later batch: 1,120 assessments, July to December
  2025, 420 from Central. Flagged: 63.9% of the batch (46.3% on validation); Central 100%, other sectors 38% to
  46%. Central's walk mean 9.56 s against 6.00 in training (ratio 1.59); age ratio 1.07. Planted: the Central team
  timed the walk over six metres. After multiplying by 4/6: Central flagged 51.7%, recall 0.773, precision 0.235.
  Central's own AUC 0.724 before and after; the whole batch's AUC 0.657 before, 0.729 after.
- **4.3 (graded)** Test period at 0.13: women 836 assessments, recall 0.801, false positive rate 0.430; men 600,
  recall 0.586, false positive rate 0.333. Counts: women 117 caught, 297 needless, 29 missed, 393 right; men 58,
  167, 41, 334. At women's recall about 21 more male fallers would be caught. Recorded prior falls of clients who
  fell: women 0.93, men 0.26. Planted: men's past falls are recorded less often. `nurse_preference` predicts sex
  with accuracy 0.913 (the larger group alone: 0.582).
- **4.4** Admission screen: training 1,829 (base 0.124), validation 592, test 579. The starter's validation AUC:
  logistic regression 0.949, tree 0.920. Planted: `discharge_letter_received`, alone AUC 0.937. Without it:
  logistic regression 0.706, tree 0.648. No repeated `record_id`. Costs 8 to 1: the rule gives 1/9 = 0.111; the
  cheapest threshold on validation for a logistic regression on the seven honest numeric columns is 0.14.
  Planted for the model card: men's emergency visits are recorded less often, as in the fall screen.
