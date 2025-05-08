from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Self

from src.setting.settings import setup_logger
from src.utils.decorators import start_finish_function

logger = setup_logger()


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self

    @start_finish_function
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features].copy()


class CategoricalTypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], columns_type: str | object = "category"):
        self.columns = columns
        self.columns_type = columns_type

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self

    @start_finish_function
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        not_found_features = []
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype(self.columns_type)
            else:
                not_found_features.append(col)

        if not_found_features:
            logger.error(f"Не найдены категориальные фичи: {not_found_features}")
        return X
