import os.path
import pandas as pd

from src.data_loader.raw_data_loader import get_df
from src.model_selection.evaluate import DualThresholdClassifier
from src.model_selection.search import search_best_model
from src.pipeline.data_preparation_pipeline import TrainTestPreparer
from src.pipeline.model_pipeline import XGBoostPipelineBuilder, LogistRegPipeline, BasePipeBuilder
from src.setting.settings import setup_logger, dbm
from src.utils.decorators import start_finish_function

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, f1_score, average_precision_score

from src.utils.io import JoblibModelIO, JsonAppendLogger

logger = setup_logger()


class DeliveryPredictionsLate:
    def __init__(self,
                 builder_pipe_cls: BasePipeBuilder,
                 param_grid: dict,
                 features_name: dict[str, list[str]],
                 n_splits: int = 3,
                 path_read_model: str = "artifacts/models/xgb_pipeline.pkl",
                 prefix: str = "xgb",
                 read_model: bool = False):
        self.path_read_model = path_read_model
        self.prefix = prefix
        self.features_name = features_name
        self.param_grid = param_grid
        self.builder_pipe_cls = builder_pipe_cls
        self.tss = TimeSeriesSplit(n_splits=n_splits)
        self.jm_io = JoblibModelIO(self.path_read_model)
        self.df: pd.DataFrame = pd.DataFrame()

        self._load_data()

        if read_model and os.path.isfile(self.jm_io.path):
            self.best_estimator = self.jm_io.load_model()
        else:
            self.best_estimator = self._build_pipeline()

        self.results_metrics: dict

    @start_finish_function
    def _load_data(self) -> None:
        engine = dbm.get_engine()
        df = get_df(engine)
        self.df = df[df.date_create >= (df.date_create.max() - pd.DateOffset(years=3))]

        feature_name = [
            'delivery_point', 'rasstoyanie', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki',
            'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'date_create', 'tonnazh', 'obem_znt',
            'kolvo_gruzovykh_mest', 'lt_stoimost_perevozki'
        ]
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_subset, self.y_subset = \
            TrainTestPreparer(feature_columns=feature_name).prepare(self.df)

    @start_finish_function
    def _build_pipeline(self):
        builder_args = self.builder_pipe_cls.__init__.__code__.co_varnames

        builder_kwargs = {
            arg: self.features_name[arg]
            for arg in builder_args
            if arg in self.features_name
        }
        builder = self.builder_pipe_cls(**builder_kwargs)
        return builder.build()

    @start_finish_function
    def _update_pipeline_with_loaded_model(self) -> None:
        pipeline = self._build_pipeline()
        loaded_model = self.best_estimator.named_steps["model"]
        self.best_estimator = pipeline.set_params(model=loaded_model)

    def _run_grid_search(self) -> None:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'ap': make_scorer(average_precision_score),
        }
        search = search_best_model(
            self.best_estimator,
            self.param_grid,
            scoring,
            self.x_subset,
            self.y_subset,
            refit_metric='ap',
            spliter=self.tss
        )
        self.best_estimator = search.best_estimator_

    def _log_metrics_and_params(self, results: dict) -> None:
        JsonAppendLogger("artifacts/params.json", key_name="params", prefix=self.prefix)\
            .append(self.best_estimator.named_steps['model'].get_params())
        JsonAppendLogger("artifacts/metrics.json", key_name="metrics", prefix=self.prefix)\
            .append(results)

    @start_finish_function
    def start_program(self,
                      features_name: dict[str, list[str]] | None = None,
                      rebuild_pipe: bool = False,
                      save_model: bool = True,
                      select_hyperparam: bool = True,
                      get_predict: bool = False):
        if features_name and rebuild_pipe:
            logger.info('Новые фичи')
            self.features_name = features_name
            self._update_pipeline_with_loaded_model()

        if select_hyperparam:
            logger.info('Подбор параметров')
            self._run_grid_search()

        if save_model:
            logger.info('Сохранение модели')
            self.jm_io.save_model(self.best_estimator)

        return self._evaluate_model(get_predict=get_predict)

    @start_finish_function
    def _evaluate_model(self, get_predict: bool = False):
        dtc = DualThresholdClassifier(
            base_estimator=self.best_estimator,
            splitter=self.tss,
            precision_cutoff=0.7,
            recall_cutoff=0.80,
        )
        dtc.fit(self.x_train, self.y_train)

        self.results_metrics = {
            'features': self.features_name,
            'cv_metrics': dtc.cv_report(),
            'test': dtc.evaluate(self.x_test, self.y_test)
        }

        self._log_metrics_and_params(self.results_metrics)

        if get_predict:
            logger.info('Предсказания модели')
            return dtc.predict(self.x_test)
        return self.results_metrics


# Подбор параметров
@start_finish_function
def boost_start(select_hyperparam: bool = True,
                read_model: bool = False,
                save_model=True,
                get_predict: bool = False,
                path_read_model: str = "artifacts/models/xgb_pipeline.pkl",
                prefix="xgb"):
    features = ['delivery_point', 'month',
                'geo_rasstoyanie_km', 'tonnazh_group', 'distance_group',
                'rasstoyanie', 'tonnazh', 'planned_delivery_days',
                'lat_zagruzki', 'lng_zagruzki', 'lat_vygruzki', 'lng_vygruzki']

    categorical_features = []
    features_name = {
        "categorical_features": categorical_features,
        "numeric_features": [],
        "other_features": [],
        "features": features
    }
    param_grid = {
        "model__n_estimators": [300],
        "model__max_depth": [3],
        "model__learning_rate": [0.1],
        # "model__min_child_weight": [1, 5],
        "model__subsample": [1.0],
        "model__colsample_bytree": [0.8],
        # "model__gamma": [0, 1],
        "model__reg_alpha": [0],
        "model__reg_lambda": [1]
    }

    # param_grid = {
    #     "model__n_estimators": [100, 300],
    #     "model__max_depth": [3, 6],
    #     "model__learning_rate": [0.05, 0.1],
    #     # "model__min_child_weight": [1, 5],
    #     "model__subsample": [0.8, 1.0],
    #     "model__colsample_bytree": [0.8, 1.0],
    #     # "model__gamma": [0, 1],
    #     "model__reg_alpha": [0, 0.1],
    #     "model__reg_lambda": [1, 5]
    # }

    dpl = DeliveryPredictionsLate(builder_pipe_cls=XGBoostPipelineBuilder,
                                  param_grid=param_grid,
                                  features_name=features_name,
                                  n_splits=3,
                                  path_read_model=path_read_model,
                                  prefix=prefix,
                                  read_model=read_model
                                  )
    # Подобрать параметры и сохранить модель с метриками
    dpl.start_program(features_name=features_name,
                      save_model=save_model,
                      select_hyperparam=select_hyperparam,
                      get_predict=get_predict)
    return dpl


@start_finish_function
def log_reg_start(select_hyperparam: bool = True,
                  read_model: bool = False,
                  save_model=True,
                  get_predict: bool = False,
                  path_read_model: str = "artifacts/models/log_reg_pipeline.pkl",
                  prefix="log_reg"):
    features = ['delivery_point', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki',
                'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'month', 'planned_delivery_days',
                'geo_rasstoyanie_km', 'distance_group', 'tonnazh_group', 'cost_group',
                'rasstoyanie', 'tonnazh', 'lt_stoimost_perevozki']

    categorical_features = ['distance_group', 'cost_group', 'region_zagruzki', 'region_vygruzki']
    other_features = ['month']

    # numeric_features = list(
    #     set(x_train[features].select_dtypes(exclude=object).columns) - set(categorical_features) - set(other_features))

    features_name = {
        "categorical": categorical_features,
        # "numeric": numeric_features,
        "other": other_features,
        "all": features
    }

    param_grid = {
        "model__penalty": ['l1', 'l2', 'elasticnet', None],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ['saga'],
        "model__class_weight": [None, 'balanced']
    }

    dpl = DeliveryPredictionsLate(builder_pipe_cls=LogistRegPipeline,
                                  param_grid=param_grid,
                                  features_name=features_name,
                                  n_splits=3,
                                  path_read_model=path_read_model,
                                  prefix=prefix,
                                  read_model=read_model
                                  )
    # Подобрать параметры и сохранить модель с метриками
    dpl.start_program(features_name=features_name,
                      save_model=save_model,
                      select_hyperparam=select_hyperparam,
                      get_predict=get_predict)
    return dpl


if __name__ == '__main__':
    boost_start(select_hyperparam=True,
                read_model=False,
                save_model=True,
                path_read_model="artifacts/models/xgb_pipeline.pkl",
                prefix="xgb")