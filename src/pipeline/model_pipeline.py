from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.features.column_transformers import FeatureSelection, CategoricalTypeCaster
from src.setting.settings import setup_logger
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils.decorators import start_finish_function

logger = setup_logger()


class BasePipeBuilder(ABC):
    @abstractmethod
    def build(self, *args, **kwargs):
        logger.error(f"Ошибка: метод build() не реализован в {self.__class__.__name__}")


class XGBoostPipelineBuilder(BasePipeBuilder):
    def __init__(self, features: list[str], categorical_features: list[str]):
        self.features = features
        self.categorical_features = categorical_features

    @start_finish_function
    def build(self, **kwargs):
        return Pipeline(steps=[
            ('cast_category', CategoricalTypeCaster(self.categorical_features)),
            ('selection', FeatureSelection(self.features)),
            ('model', XGBClassifier(enable_categorical=True, **kwargs))
        ])


class LogistRegPipeline(BasePipeBuilder):
    def __init__(self, features: list[str], categorical_columns: list[str], numeric_features: list[str],
                 other_features: list[str] = None):
        self.features = features
        self.categorical_columns = categorical_columns
        self.numeric_features = numeric_features

        logger.debug(f"Используемые колонки: {self.features}")

        if other_features is None:
            self.other_features = list(set(features) - set(categorical_columns) - set(numeric_features))
        else:
            self.other_features = other_features

    def build(self, **kwargs):
        return Pipeline(steps=[
            ('transform', self._transformers()),
            ('model', LogisticRegression(penalty=None, **kwargs))
            ])

    def _transformers(self):
        return ColumnTransformer(transformers=[
                ("numerical", StandardScaler(), self.numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_columns),
                ("other", "passthrough", self.other_features)
        ])
