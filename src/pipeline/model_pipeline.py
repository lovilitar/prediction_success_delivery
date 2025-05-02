from src.features.column_transformers import FeatureSelection, CategoricalTypeCaster
from src.setting.settings import setup_logger
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.utils.decorators import start_finish_function

logger = setup_logger()


class XGBoostPipelineBuilder:
    def __init__(self, features: list[str], categorical_columns: list[str]):
        self.features = features
        self.categorical_columns = categorical_columns

    @start_finish_function
    def build(self):
        return Pipeline(steps=[
            ('cast_category', CategoricalTypeCaster(self.categorical_columns)),
            ('selection', FeatureSelection(self.features)),
            ('model', xgb.XGBClassifier(enable_categorical=True))
        ])
