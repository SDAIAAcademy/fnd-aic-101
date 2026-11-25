# Columns of the assessment export (Hurbuth Home Care, synthetic)

The export a former analyst used for the fall screen. One row per home assessment, as the assessment system
exported it.

| Column | What it holds | When it is recorded |
|---|---|---|
| assessment_id | the assessment's code | at the visit |
| client_id | the client's code | at the visit |
| assessment_date | the day of the visit | at the visit |
| sector | the city sector of the client's home | at the visit |
| sex | F or M | at registration |
| age | age in years | at the visit |
| lives_alone | 1 if the client lives alone | at the visit |
| n_medications | medicines taken daily | at the visit |
| prior_falls_12m | falls in the year before the visit, as recorded by the nurse | at the visit |
| walk_seconds | seconds to walk four metres | at the visit |
| vision_problem | 1 if the nurse notes a vision problem | at the visit |
| uses_walking_aid | 1 if the client uses a walking aid | at the visit |
| stairs_at_home | 1 if the home has stairs the client uses | at the visit |
| diabetes | 1 if the client has diabetes | at the visit |
| bmi | body mass index, blank when not measured | at the visit |
| nurse_preference | the nurse the client asked for: female_nurse, male_nurse, no_preference | at registration |
| visit_minutes | length of the visit | at the end of the visit |
| physio_referral | 1 if the client was referred to physiotherapy | updated by the physiotherapy team during the six months after the visit |
| client_falls_total | total of prior_falls_12m over all the client's assessments in the export | computed when the export was made |
| fell_6m | 1 if the client fell in the 182 days after the visit (the label) | six months after the visit |
