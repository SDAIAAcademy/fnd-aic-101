# Follow-up task: the pressure-injury screen

About 60 minutes, on your own, within two weeks of the capstone day. It is tracked, not graded.

Repeat the capstone's workflow on a second screen of Hurbuth Home Care Company (fictional): flag clients at risk of
a **pressure injury within 90 days** of a home assessment, so that a nurse checks their skin and their mattress at
the next visit. The data are synthetic and written by `labs/prepare_data.py` with a fixed seed.

## The data: `pressure_injury.csv`

2,400 home assessments, January 2023 to June 2025.

| Column | What it holds | When it is recorded |
|---|---|---|
| record_id | the assessment's code | at the visit |
| client_id | the client's code | at the visit |
| assessment_date | the day of the visit | at the visit |
| sector | the city sector of the client's home | at the visit |
| sex | F or M | at registration |
| age | age in years | at the visit |
| lives_alone | 1 if the client lives alone | at the visit |
| mobility_score | 1 (bedbound) to 4 (walks without help) | at the visit |
| incontinence | 1 if the nurse notes incontinence | at the visit |
| diabetes | 1 if the client has diabetes | at the visit |
| nurse_preference | the nurse the client asked for | at registration |
| bmi | body mass index, blank when not measured | at the visit |
| wound_dressing_ordered | 1 if wound dressings were ordered for the client | updated when an order is placed |
| pressure_injury_90d | 1 if a pressure injury was recorded in the 90 days after the visit (the label) | 90 days after the visit |

A missed pressure injury costs about **five times** a needless skin check.

## What to do

In the same capstone repository, add a second notebook that runs end to end and contains the split by date, a
baseline and two models in pipelines, the leak check, the threshold from the costs and base rate, and a short
model card with metrics by sex and their counts. Commit as you go, then send the trainer the link to the commit.
