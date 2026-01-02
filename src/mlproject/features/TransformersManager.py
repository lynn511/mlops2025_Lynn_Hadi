from sklearn.feature_extraction import DictVectorizer
from pathlib import Path
import pickle

class TransformersManager:
    """
    Handles feature encoding and persistence using DictVectorizer
    """

    def __init__(self):
        self.dv = DictVectorizer(sparse=True)

    def fit(self, df, categorical_cols, numerical_cols):
        df = df.copy()
        df[categorical_cols] = df[categorical_cols].astype(str)

        feature_dicts = df[categorical_cols + numerical_cols].to_dict(
            orient="records"
        )

        X = self.dv.fit_transform(feature_dicts)
        return X

    def transform(self, df, categorical_cols, numerical_cols):
        df = df.copy()
        df[categorical_cols] = df[categorical_cols].astype(str)

        feature_dicts = df[categorical_cols + numerical_cols].to_dict(
            orient="records"
        )

        return self.dv.transform(feature_dicts)

    def save(self, output_dir="src/mlproject/data/transformers"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(Path(output_dir) / "dict_vectorizer.pkl", "wb") as f:
            pickle.dump(self.dv, f)

    @staticmethod
    def load(path="src/mlproject/data/transformers/dict_vectorizer.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
