from dataclasses import dataclass
from typing import Callable, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted

from src.setting.settings import setup_logger
from src.utils.decorators import start_finish_function

logger = setup_logger()

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(slots=True)
class CVResult:
    """Хранит все метрики, полученные на кросс‑валидации."""

    per_fold: dict[str, np.ndarray]
    mean: dict[str, np.ndarray]
    ap_per_fold: np.ndarray

    @property
    def ap_mean(self) -> float:
        return float(np.mean(self.ap_per_fold))


class MetricCalculator:
    """Считает метрики для набора порогов."""

    DEFAULT_METRICS: dict[str, MetricFn] = {
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
    }

    def __init__(
        self,
        thresholds: np.ndarray,
        metric_fns: Optional[dict[str, MetricFn]] = None,
    ) -> None:
        self.thresholds = thresholds
        self.metric_fns = metric_fns or self.DEFAULT_METRICS

    def __call__(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, np.ndarray]:
        results = {name: [] for name in self.metric_fns}
        for thr in self.thresholds:
            y_pred = (y_prob >= thr).astype(int)
            for name, fn in self.metric_fns.items():
                results[name].append(fn(y_true, y_pred))
        return {k: np.asarray(v) for k, v in results.items()}


@dataclass(slots=True)
class ThresholdSelector:
    """Выбор оптимальных порогов по средним метрикам CV."""
    thresholds: np.ndarray
    optimize_metric: str = "f1"
    precision_cutoff: float = 0.7
    recall_cutoff: float = 0.8

    def _best_index(self, metric_values: np.ndarray, mask: np.ndarray) -> Optional[int]:
        if np.any(mask):
            idx_local = np.argmax(metric_values[mask])
            return int(np.where(mask)[0][idx_local])
        return None

    @start_finish_function
    def select(self, mean_metrics: dict[str, np.ndarray]) -> dict[str, float | None]:
        """Возвращает словарь с выбранными порогами."""
        optimize_values = mean_metrics[self.optimize_metric]
        idx_opt = int(np.argmax(optimize_values))

        idx_late = self._best_index(
            optimize_values, mean_metrics["precision"] >= self.precision_cutoff
        )
        idx_on_time = self._best_index(
            optimize_values, mean_metrics["recall"] >= self.recall_cutoff
        )

        return {
            "optimal": float(self.thresholds[idx_opt]),
            "late": float(self.thresholds[idx_late]) if idx_late is not None else None,
            "on_time": float(self.thresholds[idx_on_time]) if idx_on_time is not None else None,
        }


class DualThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Класс‑обёртка для двухпороговой схемы решений.

    Attributes
    ----------
    base_estimator : BaseEstimator
        Любая sklearn‑совместимая модель с `fit`/`predict_proba`.
    thresholds_ : np.ndarray
        Сетка порогов (устанавливается в __init__).
    cv_result_ : CVResult
        Метрики CV после вызова `fit`.
    chosen_thresholds_ : Dict[str, float | None]
        'optimal', 'late', 'on_time'.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        splitter: TimeSeriesSplit | None = None,
        thresholds: Optional[np.ndarray] = None,
        precision_cutoff: float = 0.7,
        recall_cutoff: float = 0.8,
        metric_fns: Optional[dict[str, MetricFn]] = None,
        optimize_metric: str = "f1",
        fit_full: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.splitter = splitter or TimeSeriesSplit(n_splits=3)
        self.thresholds = thresholds or np.linspace(0.1, 0.9, 81)
        self.precision_cutoff = precision_cutoff
        self.recall_cutoff = recall_cutoff
        self.metric_fns = metric_fns
        self.optimize_metric = optimize_metric
        self.fit_full = fit_full
        self.random_state = random_state

        # Эти поля появятся после .fit()
        self.calculator_: MetricCalculator
        self.cv_result_: CVResult
        self.chosen_thresholds_: dict[str, float | None]
        self.final_model_: BaseEstimator

    def _check_fitted(self) -> None:
        check_is_fitted(self, ["final_model_", "chosen_thresholds_"])

    def _group_for_score(self, prob: float) -> str:
        """late / on_time / uncertain по выбранным порогам."""
        thr_late = self.chosen_thresholds_.get("late")
        thr_on_time = self.chosen_thresholds_.get("on_time")
        assert (thr_late is not None and thr_on_time is not None), "Пороги ещё не выбраны"
        if prob >= thr_late:
            return "late"
        if prob <= thr_on_time:
            return "on_time"
        return "uncertain"

    @start_finish_function
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DualThresholdClassifier":
        """CV‑подбор порогов + (опционально) дообучение на всём трейне."""
        X_np, y_np = X.copy(), y.copy()
        n_folds = self.splitter.n_splits
        n_thr = len(self.thresholds)

        self.calculator_ = MetricCalculator(
            thresholds=self.thresholds, metric_fns=self.metric_fns
        )

        per_fold = {k: np.zeros((n_folds, n_thr)) for k in self.calculator_.metric_fns}
        ap_scores = np.zeros(n_folds)

        for i, (tr_idx, val_idx) in enumerate(self.splitter.split(X_np)):
            mdl = clone(self.base_estimator)
            mdl.fit(X_np.iloc[tr_idx], y_np.iloc[tr_idx])

            y_prob = mdl.predict_proba(X_np.iloc[val_idx])[:, 1]
            y_true = y_np.iloc[val_idx]

            fold_metrics = self.calculator_(y_true, y_prob)
            for k in per_fold:
                per_fold[k][i] = fold_metrics[k]
            ap_scores[i] = average_precision_score(y_true, y_prob)

        mean_metrics = {k: v.mean(axis=0) for k, v in per_fold.items()}
        self.cv_result_ = CVResult(per_fold, mean_metrics, ap_scores)

        selector = ThresholdSelector(
            thresholds=self.thresholds,
            optimize_metric=self.optimize_metric,
            precision_cutoff=self.precision_cutoff,
            recall_cutoff=self.recall_cutoff,
        )
        self.chosen_thresholds_ = selector.select(self.cv_result_.mean)

        logger.info("Пороги выбраны: %s", self.chosen_thresholds_)

        if self.fit_full:
            logger.info("Дообучение финальной модели на всём трейне…")
            self.final_model_ = clone(self.base_estimator)
            self.final_model_.fit(X_np, y_np)
        else:
            self.final_model_ = mdl  # последний из CV

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.final_model_.predict_proba(X)[:, 1]

    @start_finish_function
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает метки групп ('late' / 'on_time' / 'uncertain')."""
        probs = self.predict_proba(X)
        return np.vectorize(self._group_for_score)(probs)

    def _metrics_at(self,
                    metrics: dict[str, np.ndarray],
                    threshold: float | None) -> dict[str, np.float64]:
        """Возвратить precision/recall/f1 для конкретного threshold."""
        if threshold is None:
            return {}
        idx = int(np.where(self.thresholds == threshold)[0][0])
        return {m: metrics[m][idx] for m in metrics}

    def _build_report(self,
                      metrics_dict: dict[str, np.ndarray],
                      ap_value: float | np.float64) -> dict[str, dict[str, np.float64] | np.float64]:
        return {
            'thresholds': self.chosen_thresholds_,
            'optimal_metrics': self._metrics_at(metrics_dict, self.chosen_thresholds_['optimal']),
            'late_metrics': self._metrics_at(metrics_dict, self.chosen_thresholds_.get('late', None)),
            'on_time_metrics': self._metrics_at(metrics_dict, self.chosen_thresholds_.get('on_time', None)),
            'ap': np.float64(ap_value),
        }

    @start_finish_function
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        at_thresholds: Optional[Iterable[float]] = None,
    ) -> dict[str, dict[str, float]]:
        """Метрики на произвольном наборе данных."""
        self._check_fitted()
        probs = self.predict_proba(X)
        metrics = self.calculator_(y.to_numpy(), probs)
        ap = average_precision_score(y.to_numpy(), probs)
        return self._build_report(metrics, ap)

    @start_finish_function
    def cv_report(self) -> dict[str, dict[str, np.float64] | np.float64]:
        """
        Возвращает тот же формат, но использует усреднённые по фолдам
        метрики (`self.cv_result_.mean`) и средний AP (`self.cv_result_.ap_mean`).
        Вызывать после .fit()
        """
        self._check_fitted()
        return self._build_report(self.cv_result_.mean, self.cv_result_.ap_mean)
