# Model card: <name of the screen>

One page. Every number comes from a run in this repository; say which notebook and which seed.
Adapted for this course from the model card sections of Mitchell et al., "Model Cards for Model Reporting" (2019).

| Field | What to write |
|---|---|
| **Model** | Family, inputs, version or commit, seed, date fitted |
| **Owner and reviewer** | Who prepared it (a role) and who reviewed it (a different role); who accepts it for use |
| **Intended use** | The decision it supports, who acts on a flag, and the threshold with the costs that set it |
| **Out of scope** | Uses it must not be put to |
| **Data** | Source, period, rows, how the split was made (by date), what was excluded and why |
| **Metrics** | Baseline and model on validation and on the test period (opened once): AUC, and at the threshold recall, precision, false positive rate and share flagged |
| **Metrics by subgroup** | The same rates by group, with the counts behind each rate |
| **Sensitive attributes and proxies** | Which attributes are not inputs, which inputs could carry them, and the evidence |
| **Known limits** | Shifts, measurement assumptions, groups with few rows, what the data cannot show |
| **Who should not rely on it** | People or situations for whom the score is weaker or not meaningful, and what to do instead |
| **Monitoring** | What is checked, how often, by whom, and what triggers a review |
