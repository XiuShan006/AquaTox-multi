# Dataset

The `aquatox` package contains the released data for the five-fold workflow. The model input table has 36,900 aggregated task-context samples from 5,684 standardized molecules.

- `data/common_intersection_model.csv` — model-ready rows with labels, molecular structures, exposure metadata, and task fields.
- `data/common_intersection.csv.gz` — aggregated labels and source traceability.
- `manifests/` — outer and inner fold assignments, sample roles, scaffold groups, and balance checks.
- `metadata/` — dataset summary, validation, and environment records.

Labels are median `log10(mg/L)` values grouped by standardized molecule, organism–endpoint task, exposure duration, effect type, and medium. The six tasks are fish, crustaceans, and algae crossed with EC50 and EC10.

The upstream records were curated from EU REACH, US EPA ECOTOX, and EFSA OpenFoodTox. Redistribution and reuse remain subject to the original provider terms.
