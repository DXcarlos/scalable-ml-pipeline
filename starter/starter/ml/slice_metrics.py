from ml.data import process_data
from ml.model import compute_model_metrics, inference


def compute_slice_metrics(data, categorical_features, model, encoder, label) -> list:
    """Compute model metris in slices of data"""

    results = []
    for feature_name in categorical_features:
        for category in data[feature_name].unique():
            category_data = data[data[feature_name] == category].reset_index(drop=True)
            X, y, _, _ = process_data(
                category_data, categorical_features=categorical_features, label="salary", training=False,
                encoder=encoder, lb=label
            )
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            metrics = {
                "feature": feature_name,
                "category": category,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta
            }
            results.append(metrics)

    return results
