import argparse
import sys
import os
import pandas as pd

# Make src visible inside processing container
sys.path.append("/opt/ml/processing/code/src")

from mlproject.preprocess.preprocessor import Preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_test", required=True)
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

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    pre = Preprocess()

    train_df = pre.remove_nulls(train_df)
    train_df = pre.remove_invalid_passengers(train_df)
    train_df = pre.add_trip_duration_minutes(train_df)
    train_df = pre.remove_duration_outliers(train_df)

    test_df = pre.remove_nulls(test_df)
    test_df = pre.remove_invalid_passengers(test_df)

    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test), exist_ok=True)

    train_df.to_csv(args.output_train, index=False)
    test_df.to_csv(args.output_test, index=False)


if __name__ == "__main__":
    main()
