# ICU In-Hospital Mortality Risk (MIMIC-IV)

## Project Summary

This project predicts **in-hospital mortality** for ICU patients using only the **first 24 hours**
of their stay — demographics, comorbidity burden, and early vitals/labs — the same window a
clinical early-warning score (SAPS, APACHE) would use. It's built on **MIMIC-IV**, a real,
de-identified electronic health record dataset from Beth Israel Deaconess Medical Center, published
via [PhysioNet](https://physionet.org/).

This is the patient-level companion to
[Predicting Opioid Overdose Mortality](https://github.com/Vaishnavi-K23/Drug-addiction-analysis),
which analyzes population-level CDC surveillance data. Together they cover both ends of
healthcare data science: public-health surveillance at the population level, and individual
clinical risk prediction at the patient level.

---

## Data & Access

**Dataset:** [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/) —
a 100-patient, fully de-identified subset of MIMIC-IV, freely downloadable after a quick
click-through data use agreement (no credentialing committee wait).

To reproduce this project:

1. Create a free account at [physionet.org](https://physionet.org/register/).
2. Go to the [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/)
   page and accept its data use agreement.
3. Download the `hosp/` and `icu/` folders and place them at:
   ```
   data/mimic-iv-demo/hosp/
   data/mimic-iv-demo/icu/
   ```
4. `pip install -r requirements.txt`
5. `python src/pipeline.py` — builds the cohort and feature matrix into `data/processed/`
6. Run the notebooks in `notebooks/`, in order.

**Raw MIMIC data is never committed to this repo** (`data/` is git-ignored) — the PhysioNet Data
Use Agreement prohibits redistributing the data outside PhysioNet, so every collaborator has to
go through steps 1–3 themselves.

`src/config.py` centralizes the data path, so if/when full credentialed MIMIC-IV access is
approved (requires the CITI "Data or Specimens Only Research" course + a PhysioNet DUA + committee
review — days to a few weeks), the same pipeline reruns at full scale (tens of thousands of
patients instead of 100) with a one-line path change, for properly powered results.

---

## Privacy & HIPAA

**Why this data can be used without a HIPAA Business Associate Agreement:** MIMIC-IV is
de-identified under the HIPAA **Safe Harbor** method — all 18 HIPAA identifiers are removed:
names, exact dates (shifted per-patient into a random future date range while preserving the
*interval* between events, so time-since-admission calculations stay valid), ages over 89
aggregated into a single "90+" bucket, and free-text notes scrubbed of identifying information.
Because no data element can reasonably be used to re-identify a patient, it falls outside HIPAA's
definition of Protected Health Information — which is why it's usable for research and portfolio
work without a covered entity, a BAA, or IRB review for secondary analysis.

**If this were live production PHI instead of de-identified research data**, the operational
requirements would be substantially different:

- **Minimum necessary access** — this pipeline pulls full vitals/labs/diagnoses tables; a
  production system would scope queries to only the fields a given use case needs.
- **Role-based access control** — data scientists, clinicians, and administrative staff would have
  different access tiers, enforced at the database layer, not just by convention.
- **Audit logging** — every query against PHI needs to be logged and attributable to a user, for
  HIPAA's required audit trail.
- **Encryption at rest and in transit**, for the database and for any exported files or model
  artifacts that could indirectly leak patient information.
- **A signed Business Associate Agreement** with any cloud vendor (compute, storage, MLOps
  tooling) that touches PHI, per HIPAA's BAA requirement for third parties handling covered data.

**PhysioNet Data Use Agreement compliance:** raw MIMIC files are excluded from version control via
`.gitignore`; only code, methodology, and (once run) locally-generated derived metrics/plots are
part of this repo.

---

## Cohort

Built in [`src/cohort.py`](src/cohort.py) from `icustays` + `admissions` + `patients`:

- First ICU stay per patient only (avoids leaking outcome information across a patient's multiple
  admissions into what should be an independent-patient prediction)
- Age ≥ 18
- ICU stay length ≥ 4 hours (need enough time to actually observe a 24h window)
- Has a recorded outcome label

Every step prints a row-count waterfall — standard practice in clinical study design, so the
cohort's provenance is auditable rather than an opaque filtered dataframe.

## Features

Built in [`src/features.py`](src/features.py), all restricted to the **first 24 hours** of the ICU
stay (no information from later in the stay — that would leak the future into a prediction that's
supposed to be made early):

- Demographics: age, gender, admission type/location, insurance
- Vitals (heart rate, respiratory rate, SpO2, blood pressure, temperature, GCS components):
  min/max/mean over the window
- Labs (creatinine, WBC, lactate, glucose, sodium, potassium, hemoglobin): min/max/mean
- **A `_missing` indicator alongside every vital/lab** — a test that wasn't ordered is itself
  clinically informative, not just a gap to impute away
- Comorbidity flags (CHF, diabetes, renal failure, liver disease, COPD, hypertension, malignancy)
  from ICD-9/ICD-10 diagnosis codes, plus a summary `comorbidity_score`. This uses a small curated
  set of code prefixes, not the full validated AHRQ Elixhauser mapping — a deliberate scoping
  choice for a 100-patient demo, noted here rather than overclaiming rigor. At full-scale MIMIC-IV
  this should be swapped for a validated implementation (e.g. AHRQ's reference mapping or the
  `comorbidipy` package).

## Modeling

`notebooks/02_Modeling.ipynb`: Logistic Regression baseline (interpretable, in the spirit of
clinically-trusted scoring systems) → Random Forest → XGBoost. Evaluated on **AUROC and AUPRC**,
not accuracy — mortality is a rare/imbalanced outcome, so a model that always predicts "survived"
would score high accuracy while being clinically useless. Includes a calibration curve (a risk
score clinicians act on needs to mean what it says, not just rank patients correctly) and SHAP
explainability, checked against clinical plausibility rather than taken at face value.

## Fairness / Subgroup Audit

`notebooks/03_Fairness_Audit.ipynb` recomputes AUROC/AUPRC and calibration split by race and
gender — healthcare AI has a real history of subgroup failures (pulse-oximetry bias by skin tone,
race-adjusted eGFR formulas), so this is a habit, not a one-off check. **With only 100 demo
patients this audit is underpowered** — that's stated explicitly rather than treating a
small-sample difference as a finding. It's included to demonstrate the audit methodology to run at
scale, once full MIMIC-IV access is available.

---

## Project Structure

```
mimic-icu-mortality-risk/
├── src/
│   ├── config.py       # data paths, cohort thresholds
│   ├── cohort.py        # cohort construction + inclusion/exclusion waterfall
│   ├── features.py      # first-24h vitals/labs/comorbidity feature engineering
│   └── pipeline.py       # end-to-end: raw tables -> cohort -> features -> train/test split
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Fairness_Audit.ipynb
├── data/                 # git-ignored — populate yourself per the Data & Access section
└── requirements.txt
```

## Future Improvements

- Full AHRQ Elixhauser comorbidity mapping in place of the curated prefix list
- 30-day readmission as a second prediction task, once full MIMIC-IV gives enough
  multi-admission patients for the outcome to have real signal
- Time-series modeling (e.g. an RNN/transformer over the raw vitals sequence) instead of
  window-aggregated features
- Threshold/reweighting mitigation for any subgroup gap found once the fairness audit is rerun at
  full scale

## Author

Vaishnavi Kulkarni
MS Data Science
