from src.data_loader.raw_data_loader import get_df
from src.model_selection.evaluate import evaluate_on_test
from src.model_selection.search import search_best_model
from src.model_selection.threshold import ThresholdOptimizer
from src.pipeline.data_preparation_pipeline import TrainTestPreparer
from src.pipeline.model_pipeline import XGBoostPipelineBuilder
from src.setting.settings import setup_logger, dbm
from src.utils.decorators import start_finish_function

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, f1_score, average_precision_score


logger = setup_logger()


@start_finish_function
def start_program(select_hyperparam=True):
    logger.info(f"Старт программы")

    engine = dbm.get_engine()

    df = get_df(engine)
    feature_name = ['delivery_point', 'rasstoyanie', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki', 'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'date_create', 'tonnazh', 'obem_znt', 'kolvo_gruzovykh_mest', 'lt_stoimost_perevozki']

    # Разделение на тест, трейн и подбора гипер параметров
    x_train, y_train, x_test, y_test, x_subset, y_subset = TrainTestPreparer(feature_columns=feature_name).prepare(df)

    categorical_columns = ['distance_group', 'cost_group', 'region_zagruzki', 'region_vygruzki']

    features = ['delivery_point', 'region_zagruzki', 'lat_zagruzki', 'lng_zagruzki',
                'region_vygruzki', 'lat_vygruzki', 'lng_vygruzki', 'month', 'planned_delivery_days',
                'geo_rasstoyanie_km', 'distance_group', 'tonnazh_group', 'cost_group',
                'rasstoyanie', 'tonnazh', 'lt_stoimost_perevozki']

    param_grid = {
        'model__n_estimators': [100, 200],
        # 'model_boost__max_depth': [3, 5, 7],
        # 'model_boost__learning_rate': [0.01, 0.1, 0.3],
        # 'model_boost__subsample': [0.8, 1.0],
        # 'model_boost__colsample_bytree': [0.8, 1.0]
    }

    param_scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'ap': make_scorer(average_precision_score)
    }

    tss = TimeSeriesSplit(n_splits=3)

    pipe_xgboost = XGBoostPipelineBuilder(features, categorical_columns).build()

    if select_hyperparam:
        search = search_best_model(pipe_xgboost, param_grid, param_scoring, x_subset, y_subset, refit_metric='ap')
        best_model = search.best_estimator_
    else:
        best_model = pipe_xgboost

     # Подбор порога отсечения
    optimizer = ThresholdOptimizer(best_model)
    optimal_threshold, train_metrics = optimizer.optimize(x_train, y_train, tss)

    # Финальная проверка на тесте
    test_metrics = evaluate_on_test(best_model, x_test, y_test, optimal_threshold)

    # Итог
    results = {
        'threshold': optimal_threshold,
        'train': train_metrics,
        'test': test_metrics
    }

    logger.info(results)


if __name__ == '__main__':
    start_program(select_hyperparam=True)
