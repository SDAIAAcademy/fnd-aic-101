---
course: "How Machine Learning Works"
code: "FND-AIC-101"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-aic-101/slides/day4.html"
---

# How Machine Learning Works

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
