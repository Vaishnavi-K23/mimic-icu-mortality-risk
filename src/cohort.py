"""
Cohort construction: first ICU stay per patient, with documented
inclusion/exclusion criteria (a row-count "waterfall") in the style of a
clinical study, not just an unexplained filter.
"""

import pandas as pd

from config import HOSP_DIR, ICU_DIR, MIN_AGE, MIN_ICU_STAY_HOURS


def _load_raw():
    patients = pd.read_csv(HOSP_DIR / "patients.csv.gz")
    admissions = pd.read_csv(
        HOSP_DIR / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icustays = pd.read_csv(
        ICU_DIR / "icustays.csv.gz", parse_dates=["intime", "outtime"]
    )
    return patients, admissions, icustays


def build_cohort(verbose: bool = True) -> pd.DataFrame:
    patients, admissions, icustays = _load_raw()
    waterfall = [("all ICU stays", len(icustays))]

    df = icustays.merge(
        admissions[
            [
                "subject_id",
                "hadm_id",
                "admission_type",
                "admission_location",
                "insurance",
                "race",
                "hospital_expire_flag",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="left",
    ).merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )

    # Age at the time of this ICU stay (MIMIC-IV's standard age calc:
    # anchor_age was the patient's age during anchor_year).
    df["age"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])

    # First ICU stay per patient only, to avoid leaking outcome information
    # from prior stays into what should be an independent-patient prediction.
    df = df.sort_values("intime").groupby("subject_id", as_index=False).first()
    waterfall.append(("first ICU stay per patient", len(df)))

    df = df[df["age"] >= MIN_AGE]
    waterfall.append((f"age >= {MIN_AGE}", len(df)))

    df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    df = df[df["los_hours"] >= MIN_ICU_STAY_HOURS]
    waterfall.append((f"ICU stay length >= {MIN_ICU_STAY_HOURS}h", len(df)))

    df = df.dropna(subset=["hospital_expire_flag"])
    waterfall.append(("has outcome label", len(df)))

    if verbose:
        print("Cohort construction waterfall:")
        for step, n in waterfall:
            print(f"  {step:<35} n={n}")

    keep_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "los_hours",
        "age",
        "gender",
        "race",
        "admission_type",
        "admission_location",
        "insurance",
        "hospital_expire_flag",
    ]
    return df[keep_cols].reset_index(drop=True)


if __name__ == "__main__":
    cohort = build_cohort()
    print(cohort.head())
