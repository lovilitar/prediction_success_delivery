from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.utils.decorators import start_finish_function


@start_finish_function
def search_best_model(
    estimator,
    param_grid: Dict[str, Any],
    scoring: Dict[str, Any],
    x,
    y,
    refit_metric: str = 'f1',
    spliter: int = TimeSeriesSplit(n_splits=3),
    n_jobs: int = -1,
    verbose: int = 10,
    return_train_score: bool = True
):

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=spliter,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=return_train_score
    )
    search.fit(x, y)
    return search
