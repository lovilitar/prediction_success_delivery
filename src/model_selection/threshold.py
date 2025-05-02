from typing import List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from src.utils.decorators import start_finish_function


class ThresholdOptimizer:
    def __init__(self, model, thresholds: np.ndarray = np.linspace(0.1, 0.9, 81)):
        self.model = model
        self.thresholds = thresholds

    @start_finish_function
    def optimize(self, x_train, y_train, splitter) -> Tuple[float, dict]:
        best_thresholds = []
        best_scores = {'precision': [], 'recall': [], 'f1': [], 'ap': []}

        for train_idx, val_idx in splitter.split(x_train):
            X_train_fold, X_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            self.model.fit(X_train_fold, y_train_fold)
            val_probs = self.model.predict_proba(X_val_fold)[:, 1]

            fold_scores = self._evaluate_thresholds(val_probs, y_val_fold)
            best_idx = np.argmax(fold_scores['f1'])

            best_thresholds.append(self.thresholds[best_idx])
            for key in best_scores:
                best_scores[key].append(fold_scores[key][best_idx])

        return np.mean(best_thresholds), {
            key: round(np.mean(vals), 4) for key, vals in best_scores.items()
        }

    def _evaluate_thresholds(self, probs, y_true) -> dict:
        scores = {'precision': [], 'recall': [], 'f1': []}
        for t in self.thresholds:
            preds = (probs >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary')
            scores['precision'].append(p)
            scores['recall'].append(r)
            scores['f1'].append(f)
        scores['ap'] = [average_precision_score(y_true, probs)] * len(self.thresholds)
        return scores
