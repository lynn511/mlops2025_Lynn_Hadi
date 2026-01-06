from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


class ModelTrainer:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return {
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
