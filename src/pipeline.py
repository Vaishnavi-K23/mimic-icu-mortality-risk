"""
End-to-end: raw MIMIC-IV tables -> cohort -> feature matrix -> train/test
split, written to data/processed/. Run this once before the notebooks.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from cohort import build_cohort
from features import build_feature_matrix
from config import PROCESSED_DIR, RANDOM_STATE

CATEGORICAL_COLS = ["gender", "race", "admission_type", "admission_location", "insurance"]
# stay_id is kept (not used as a model feature, but needed by the fairness
# audit notebook to join predictions back to race/gender). Drop it before
# fitting any model.
DROP_COLS = ["subject_id", "hadm_id", "intime", "outtime"]
TARGET_COL = "hospital_expire_flag"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cohort = build_cohort()
    df = build_feature_matrix(cohort)

    df.to_csv(PROCESSED_DIR / "feature_matrix_raw.csv", index=False)

    modeling_df = pd.get_dummies(df.drop(columns=DROP_COLS), columns=CATEGORICAL_COLS, dummy_na=True)

    train_df, test_df = train_test_split(
        modeling_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=modeling_df[TARGET_COL],
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\nCohort size: {len(df)}")
    print(f"Train/test: {len(train_df)}/{len(test_df)}")
    print(f"Mortality prevalence — train: {train_df[TARGET_COL].mean():.3f}, "
          f"test: {test_df[TARGET_COL].mean():.3f}")
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
