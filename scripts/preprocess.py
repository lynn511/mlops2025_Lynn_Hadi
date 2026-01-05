import argparse
import pandas as pd
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing step")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_train", type=str, required=True)
    parser.add_argument("--output_test", type=str, required=True)

    return parser.parse_args()


def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[df["passenger_count"] > 0]

    # add trip duration in minutes (example)
    df["trip_duration_minutes"] = df["trip_duration"] / 60.0

    # remove extreme outliers
    q_low = df["trip_duration_minutes"].quantile(0.01)
    q_high = df["trip_duration_minutes"].quantile(0.99)
    df = df[(df["trip_duration_minutes"] >= q_low) &
            (df["trip_duration_minutes"] <= q_high)]

    return df


def preprocess_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df[df["passenger_count"] > 0]
    return df


def main():
    args = parse_args()

    print("📥 Loading data")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    print("🧹 Preprocessing train data")
    train_df = preprocess_train(train_df)

    print("🧹 Preprocessing test data")
    test_df = preprocess_test(test_df)

    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test), exist_ok=True)

    train_df.to_csv(args.output_train, index=False)
    test_df.to_csv(args.output_test, index=False)

    print("✅ Preprocessing completed")
    print(f"Train saved to {args.output_train}")
    print(f"Test saved to {args.output_test}")


if __name__ == "__main__":
    main()
