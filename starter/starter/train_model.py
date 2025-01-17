# Script to train machine learning model.
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.slice_metrics import compute_slice_metrics


# Add code to load in the data.
data = pd.read_csv('./starter/data/census.csv', skipinitialspace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Test model
test_preds = inference(model, X_test)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
print(f"Precision : {precision}. Recall : {recall}. FBeta : {fbeta}")

# Compute metrics on slices of data in all categorical features slices
slice_metrics_results = compute_slice_metrics(data, cat_features, model, encoder, lb)

# Save to txt file
with open('./starter/slice_output.txt', 'w') as f:
    for metrics in slice_metrics_results:
        f.write(f"Feature : {metrics['feature']}. Category : {metrics['category']}. "
                f"Precision : {metrics['precision']}. Recall : {metrics['recall']}. FBeta : {metrics['fbeta']}\n")

# Save the model and the encoder
pickle.dump(model, open('./starter/model/model.pkl', 'wb'))
pickle.dump(encoder, open('./starter/model/encoder.pkl', 'wb'))
