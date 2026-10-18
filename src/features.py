"""
Feature engineering from the first FEATURE_WINDOW_HOURS of each ICU stay.

Design choices worth calling out (also in the README):
- Vitals/labs are summarized as min/max/mean over the window rather than
  raw time series, to keep one row per ICU stay (a tabular-model choice,
  not a sequence-model one — documented as a scoping decision).
- A missing lab/vital in the window is itself clinically informative (a
  test that wasn't ordered), so every aggregated feature gets a paired
  `<feature>_missing` indicator instead of being silently imputed away.
- Comorbidity flags use a small curated set of ICD-9/ICD-10 prefixes for
  major chronic conditions, not the full AHRQ Elixhauser mapping — a
  scoping choice for a 100-patient demo. At full MIMIC-IV scale this
  should be replaced with a validated mapping (e.g. the AHRQ SAS/R
  reference implementation, or the `comorbidipy` package).
"""

import numpy as np
import pandas as pd

from config import HOSP_DIR, ICU_DIR, FEATURE_WINDOW_HOURS

VITAL_ITEMIDS = {
    "heart_rate": [220045],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "sbp": [220179, 220050],
    "dbp": [220180, 220051],
    "temp_c": [223762],
    "temp_f": [223761],  # converted to C and merged into temp_c below
    "gcs_eye": [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
}

LAB_ITEMIDS = {
    "creatinine": [50912],
    "wbc": [51301],
    "lactate": [50813],
    "glucose": [50931],
    "sodium": [50983],
    "potassium": [50971],
    "hemoglobin": [51222],
}

# (label, [(icd_version, code_prefixes)]) — deliberately small and curated,
# see module docstring.
COMORBIDITY_RULES = {
    "chf": [(9, ["428"]), (10, ["I50"])],
    "diabetes": [(9, ["250"]), (10, ["E10", "E11"])],
    "renal_failure": [(9, ["585", "586"]), (10, ["N18", "N19"])],
    "liver_disease": [(9, ["571"]), (10, ["K70", "K71", "K72", "K73", "K74"])],
    "copd": [(9, ["496"]), (10, ["J44"])],
    "hypertension": [(9, ["401"]), (10, ["I10"])],
    "malignancy": [(9, [str(c) for c in range(140, 209)]), (10, ["C" + f"{c:02d}" for c in range(0, 98)])],
}


def _window_aggregate(events: pd.DataFrame, cohort: pd.DataFrame, itemid_map: dict, time_col: str) -> pd.DataFrame:
    """Merge events onto the cohort's per-stay time window, filter to the
    first FEATURE_WINDOW_HOURS, then pivot to one row per stay_id with
    <name>_min/_max/_mean columns plus <name>_missing indicators."""
    merged = events.merge(
        cohort[["stay_id", "intime"]], on="stay_id", how="inner"
    )
    merged["hours_from_admit"] = (
        merged[time_col] - merged["intime"]
    ).dt.total_seconds() / 3600
    merged = merged[
        (merged["hours_from_admit"] >= 0)
        & (merged["hours_from_admit"] <= FEATURE_WINDOW_HOURS)
    ]

    out = pd.DataFrame({"stay_id": cohort["stay_id"]}).set_index("stay_id")
    for name, itemids in itemid_map.items():
        sub = merged[merged["itemid"].isin(itemids)]
        n_matched = len(sub)
        print(f"  {name:<12} itemids={itemids} -> {n_matched} rows in window "
              f"(VERIFY against your d_items/d_labitems if this is 0)")
        agg = sub.groupby("stay_id")["valuenum"].agg(["min", "max", "mean"])
        agg.columns = [f"{name}_min", f"{name}_max", f"{name}_mean"]
        out = out.join(agg, how="left")
        out[f"{name}_missing"] = out[f"{name}_mean"].isna().astype(int)

    return out.reset_index()


def build_vital_features(cohort: pd.DataFrame) -> pd.DataFrame:
    chartevents = pd.read_csv(
        ICU_DIR / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    feats = _window_aggregate(chartevents, cohort, VITAL_ITEMIDS, "charttime")

    # Merge Fahrenheit into Celsius (some records only have one or the other).
    f_to_c = (feats["temp_f_mean"] - 32) * 5 / 9
    feats["temp_c_mean"] = feats["temp_c_mean"].fillna(f_to_c)
    feats["temp_c_missing"] = (
        feats["temp_c_mean"].isna().astype(int) & feats["temp_f_missing"]
    )
    feats = feats.drop(
        columns=[c for c in feats.columns if c.startswith("temp_f_")]
    )
    return feats


def build_lab_features(cohort: pd.DataFrame) -> pd.DataFrame:
    labevents = pd.read_csv(
        HOSP_DIR / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    # labevents are keyed by hadm_id, not stay_id — join through the cohort
    # so the same first-24h-of-ICU-stay window logic applies.
    labevents = labevents.merge(
        cohort[["stay_id", "hadm_id"]], on="hadm_id", how="inner"
    )
    return _window_aggregate(labevents, cohort, LAB_ITEMIDS, "charttime")


def build_comorbidity_features(cohort: pd.DataFrame) -> pd.DataFrame:
    diagnoses = pd.read_csv(
        HOSP_DIR / "diagnoses_icd.csv.gz",
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    diagnoses = diagnoses.merge(
        cohort[["stay_id", "hadm_id"]], on="hadm_id", how="inner"
    )

    n_dx = diagnoses.groupby("stay_id")["icd_code"].nunique().rename("n_diagnoses")

    flags = pd.DataFrame({"stay_id": cohort["stay_id"]}).set_index("stay_id")
    for name, version_rules in COMORBIDITY_RULES.items():
        matched_stays = set()
        for version, prefixes in version_rules:
            sub = diagnoses[diagnoses["icd_version"] == version]
            mask = sub["icd_code"].str.startswith(tuple(prefixes))
            matched_stays |= set(sub.loc[mask, "stay_id"])
        flags[name] = flags.index.isin(matched_stays).astype(int)

    flags = flags.join(n_dx, how="left")
    flags["n_diagnoses"] = flags["n_diagnoses"].fillna(0).astype(int)
    flags["comorbidity_score"] = flags[list(COMORBIDITY_RULES.keys())].sum(axis=1)
    return flags.reset_index()


def build_feature_matrix(cohort: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating vitals (first {}h)...".format(FEATURE_WINDOW_HOURS))
    vitals = build_vital_features(cohort)
    print("Aggregating labs (first {}h)...".format(FEATURE_WINDOW_HOURS))
    labs = build_lab_features(cohort)
    print("Building comorbidity flags...")
    comorbid = build_comorbidity_features(cohort)

    df = cohort.merge(vitals, on="stay_id").merge(labs, on="stay_id").merge(
        comorbid, on="stay_id"
    )
    return df


if __name__ == "__main__":
    from cohort import build_cohort

    cohort = build_cohort()
    features = build_feature_matrix(cohort)
    print(features.shape)
    print(features.head())
