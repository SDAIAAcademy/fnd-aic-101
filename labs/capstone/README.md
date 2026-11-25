# Capstone brief: the admission screen

Hurbuth Home Care Company (fictional) wants a second screen. After each home assessment, it should flag clients at
risk of an **unplanned hospital admission within 90 days**, so that a nurse makes an extra check-in visit in the
following fortnight. You build the screen, write a modelling report, and defend it in ten minutes.

The data are synthetic and written by `labs/prepare_data.py` with a fixed seed. Clients are codes; nothing
identifies a person. Only this synthetic file is used: no client data of any real organisation.

## The data: `admissions.csv`

3,000 home assessments, January 2023 to June 2025, one row each.

| Column | What it holds | When it is recorded |
|---|---|---|
| record_id | the assessment's code | at the visit |
| client_id | the client's code | at the visit |
| assessment_date | the day of the visit | at the visit |
| sector | the city sector of the client's home | at the visit |
| sex | F or M | at registration |
| age | age in years | at the visit |
| lives_alone | 1 if the client lives alone | at the visit |
| n_medications | medicines taken daily | at the visit |
| er_visits_12m | emergency department visits in the year before, as recorded by the nurse | at the visit |
| heart_failure | 1 if the client has heart failure | at the visit |
| diabetes | 1 if the client has diabetes | at the visit |
| uses_walking_aid | 1 if the client uses a walking aid | at the visit |
| nurse_preference | the nurse the client asked for: female_nurse, male_nurse, no_preference | at registration |
| bmi | body mass index, blank when not measured | at the visit |
| discharge_letter_received | 1 if a hospital discharge letter for the client reached Hurbuth | updated when a letter arrives |
| admitted_90d | 1 if the client had an unplanned admission in the 90 days after the visit (the label) | 90 days after the visit |

## The costs

The clinical director's estimate: a **missed admission costs about eight times** a needless check-in visit. Use
these costs with the base rate of the training period to choose the threshold.

## What you submit

A public GitHub repository, with its history kept, holding one notebook that runs end to end with its fixed seed
and contains:

1. The split by date: training before July 2024, validation to December 2024, test from January 2025.
2. A baseline and two models, every preprocessing step inside a pipeline fitted on training rows, compared on
   validation. The test set is opened once, for the chosen model.
3. A leak check: every column you drop, and why.
4. The threshold decision: from the costs and the base rate, with the confusion matrix and recall, precision and
   false positive rate at that threshold.
5. A one-page model card from `labs/model_card_template.md`, with metrics by sex and their counts, known limits,
   who should not rely on the screen, the owner (a role) and a different reviewer (a role).

Do not put any real person's data, or anything classified above Public, into a generative AI tool. The data here
are synthetic and public.

## The defence: ten minutes

Six minutes of talk, then four of questions. In the questions you re-run the notebook from a fresh start and
explain a result the instructor selects. AI-full is allowed for the capstone's code; you must be able to explain
every line you submit.

## Rubric

| Criterion | Weight |
|---|---|
| Held-out workflow: a baseline and two models compared on validation with a fixed seed, the test set opened once, the notebook running end to end | 25 |
| Leakage-safe data handling: the split by date, the field recorded after the outcome removed, every preprocessing step fitted on training rows inside a pipeline | 15 |
| Threshold decision: chosen from the stated error costs and the base rate, with the confusion matrix and metrics at that threshold | 20 |
| Model card: intended use, data, metrics by subgroup with counts, known limits and who should not rely on the model, with an owner and a different reviewer | 20 |
| Defence: re-runs the notebook and explains the result the instructor selects | 20 |

Start from `labs/D4_4_capstone_start.ipynb` (Lab 4.4).
