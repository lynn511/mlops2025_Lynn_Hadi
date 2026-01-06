import argparse
import os
import sys

# 🔑 Make mlproject importable
sys.path.append("/opt/ml/code/src")

import pandas as pd
from mlproject.train.trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker provides this channel automatically
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load training data
    train_path = os.path.join(args.train, "train.csv")
    df = pd.read_csv(train_path)

    # Split X / y
    y = df["trip_duration"]
    X = df.drop(columns=["trip_duration"])

    # Train (linear only)
    trainer = ModelTrainer(model_type="linear")
    trainer.train(X, y)

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    trainer.save(os.path.join(args.model_dir, "model.joblib"))

    print("✅ Training completed successfully")


if __name__ == "__main__":
    main()

