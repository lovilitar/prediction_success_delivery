import os.path

import pandas as pd

from src.data_loader.raw_data_loader import get_df
from src.model_selection.evaluate import evaluate_on_test
from src.model_selection.search import search_best_model
from src.model_selection.threshold import ThresholdOptimizer
from src.pipeline.data_preparation_pipeline import TrainTestPreparer
from src.pipeline.model_pipeline import XGBoostPipelineBuilder, LogistRegPipeline
from src.setting.settings import setup_logger, dbm
from src.utils.decorators import start_finish_function

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, f1_score, average_precision_score

from src.utils.io import JoblibModelIO, JsonAppendLogger

logger = setup_logger()


@start_finish_function
def get_data():
    logger.info(f"Старт программы")

    engine = dbm.get_engine()

    df = get_df(engine)
    df = df[df.date_create >= (df.date_create.max() - pd.DateOffset(years=3))]

    feature_name = ['delivery_point', 'rasstoyanie', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki',
                    'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'date_create', 'tonnazh', 'obem_znt',
                    'kolvo_gruzovykh_mest', 'lt_stoimost_perevozki']

    # Разделение на тест, трейн и подбора гипер параметров
    x_train, y_train, x_test, y_test, x_subset, y_subset = TrainTestPreparer(feature_columns=feature_name).prepare(df)
    return x_train, y_train, x_test, y_test, x_subset, y_subset


@start_finish_function
def start_calculation_score(x_train: pd.DataFrame,
                            y_train: pd.Series,
                            x_test: pd.DataFrame,
                            y_test: pd.Series,
                            x_subset: pd.DataFrame,
                            y_subset: pd.Series,
                            pipe,
                            param_grid: dict,
                            features_name: dict = {},
                            select_hyperparam: bool = True,
                            read_model: bool = False,
                            path_read_model: str = "artifacts/models/xgb_pipeline.pkl",
                            prefix="xgb"):

    tss = TimeSeriesSplit(n_splits=3)
    param_scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'ap': make_scorer(average_precision_score)
    }

    jm_io = JoblibModelIO(path_read_model)
    if read_model and os.path.isfile(getattr(jm_io, "path", path_read_model)):
        best_model = jm_io.load_model()
    else:

        if select_hyperparam:
            search = search_best_model(pipe, param_grid, param_scoring, x_subset, y_subset, refit_metric='ap')
            best_model = search.best_estimator_
        else:
            best_model = pipe

     # Подбор порога отсечения
    optimizer = ThresholdOptimizer(best_model)
    optimal_threshold, train_metrics = optimizer.optimize(x_train, y_train, tss)

    # Финальная проверка на тесте
    test_metrics = evaluate_on_test(best_model, x_test, y_test, optimal_threshold)

    results = {
        'features': features_name,
        'threshold': optimal_threshold,
        'train': train_metrics,
        'test': test_metrics
    }

    logger.info(results)

    # Сохранение моделей и метрик
    jm_io.save_model(best_model)

    JsonAppendLogger(
        "artifacts/params.json", key_name="params", prefix=prefix
    ).append(best_model.named_steps['model'].get_params())
    JsonAppendLogger(
        "artifacts/metrics.json", key_name="metrics", prefix=prefix
    ).append(results)


@start_finish_function
def boost_start(select_hyperparam: bool = True, read_model: bool = False, path_read_model: str = "artifacts/models/xgb_pipeline.pkl", prefix="xgb"):
    x_train, y_train, x_test, y_test, x_subset, y_subset = get_data()

    features = ['delivery_point', 'lat_zagruzki', 'lng_zagruzki',
                'region_zagruzki', 'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'month', 'planned_delivery_days',
                'geo_rasstoyanie_km', 'distance_group', 'tonnazh_group', 'cost_group',
                'rasstoyanie', 'tonnazh', 'lt_stoimost_perevozki']
    features = ['delivery_point', 'month',
                'geo_rasstoyanie_km', 'tonnazh_group', 'distance_group',
                'rasstoyanie', 'tonnazh', 'planned_delivery_days',
                'lat_zagruzki', 'lng_zagruzki', 'lat_vygruzki', 'lng_vygruzki']

    categorical_features = []

    features_name = {
        "categorical": categorical_features,
        "numeric": [],
        "other": [],
        "all": features
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

    pipe_xgboost = XGBoostPipelineBuilder(features, categorical_features).build()
    start_calculation_score(x_train, y_train, x_test, y_test, x_subset, y_subset, pipe=pipe_xgboost, param_grid=param_grid, features_name=features_name, select_hyperparam=select_hyperparam, read_model=read_model, path_read_model=path_read_model, prefix=prefix)


@start_finish_function
def log_reg_start(select_hyperparam: bool = True, read_model: bool = False, path_read_model: str = "artifacts/models/log_reg_pipeline.pkl", prefix="log_reg"):
    x_train, y_train, x_test, y_test, x_subset, y_subset = get_data()
    features = ['delivery_point', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki',
                'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'month', 'planned_delivery_days',
                'geo_rasstoyanie_km', 'distance_group', 'tonnazh_group', 'cost_group',
                'rasstoyanie', 'tonnazh', 'lt_stoimost_perevozki']

    categorical_features = ['distance_group', 'cost_group', 'region_zagruzki', 'region_vygruzki']
    other_features = ['month']

    numeric_features = list(
        set(x_train[features].select_dtypes(exclude=object).columns) - set(categorical_features) - set(other_features))

    features_name = {
        "categorical": categorical_features,
        "numeric": numeric_features,
        "other": other_features,
        "all": features
    }

    param_grid = {
        "model__penalty": ['l1', 'l2', 'elasticnet', None],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ['saga'],
        "model__class_weight": [None, 'balanced']
    }

    pipe_log_reg = LogistRegPipeline(features, categorical_features, numeric_features).build()
    start_calculation_score(x_train, y_train, x_test, y_test, x_subset, y_subset, pipe=pipe_log_reg, param_grid=param_grid, features_name=features_name, select_hyperparam=select_hyperparam, read_model=read_model, path_read_model=path_read_model, prefix=prefix)


if __name__ == '__main__':
    boost_start(select_hyperparam=True, read_model=False, path_read_model="artifacts/models/xgb_pipeline.pkl", prefix="xgb")

    # log_reg_start(select_hyperparam=True, path_read_model="artifacts/models/log_reg_pipeline.pkl", prefix="log_reg")
