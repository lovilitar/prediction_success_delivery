from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from src.utils.decorators import start_finish_function


@start_finish_function
def evaluate_on_test(model, x_test, y_test, threshold: float) -> dict:
    y_probs = model.predict_proba(x_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    ap = average_precision_score(y_test, y_probs)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'ap': round(ap, 4)
    }
