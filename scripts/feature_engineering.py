import argparse

import sys
from pathlib import Path

# Go up one level from scripts to project root, then add src
sys.path.append(str(Path(__file__).resolve().parent.parent))
# OR
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlproject.features.featurizer import Featurizer
from src.mlproject.features.TransformersManager import TransformersManager
from src.mlproject.utils.dataloader import DataLoader
from src.mlproject.utils.datasaver import save_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Featurize train and test datasets, fit DictVectorizer on train only."
    )
    parser.add_argument("--train_input", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--test_input", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--train_output", type=str, required=True, help="Path to save featurized train CSV")
    parser.add_argument("--test_output", type=str, required=True, help="Path to save featurized test CSV")

    args = parser.parse_args()

    # -----------------------------
    # Load data using DataLoader
    # -----------------------------
    loader = DataLoader(train_path=args.train_input, test_path=args.test_input)
    df_train, df_test = loader.load()
    print(f"[INFO] Loaded {len(df_train)} train rows and {len(df_test)} test rows")

    # -----------------------------
    # Initialize featurizer and transformer
    # -----------------------------
    featurizer = Featurizer()
    tm = TransformersManager()

    categorical_cols = ["vendor_id", "store_and_fwd_flag"]
    numerical_cols = [
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "is_weekend",
        "trip_distance_km",
        "manhattan_distance",
        "is_night"
    ]

    # -----------------------------
    # Featurize datasets
    # -----------------------------
    df_train_feat = featurizer.featurize(df_train)
    df_test_feat = featurizer.featurize(df_test)
    print("[INFO] Applied featurize() to both train and test datasets")

    # -----------------------------
    # Fit DictVectorizer on train only
    # -----------------------------
    X_train = tm.fit(df_train_feat, categorical_cols, numerical_cols)
    tm.save()
    print("[INFO] Fitted DictVectorizer on train and saved to disk")

    # -----------------------------
    # Save featurized datasets
    # -----------------------------
    save_dataframe(df_train_feat, args.train_output)
    save_dataframe(df_test_feat, args.test_output)
    print(f"[INFO] Saved featurized train dataset to {args.train_output}")
    print(f"[INFO] Saved featurized test dataset to {args.test_output}")

    print("[INFO] Train featurization complete. Test can be transformed later with saved DictVectorizer.")



#python scripts/feature_engineering.py --train_input src/mlproject/data/processed/train_clean.csv --test_input src/mlproject/data/processed/test_clean.csv --train_output src/mlproject/data/featurized/train_features.csv --test_output src/mlproject/data/featurized/test_features.csv
