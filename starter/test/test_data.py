import pandas as pd
import numpy as np
import sklearn
import pytest

from starter.starter.ml.data import process_data


@pytest.fixture
def data():
    """Fixture to read census data """
    data = pd.read_csv('../data/census.csv', skipinitialspace=True)
    return data


@pytest.fixture
def categorical_features():
    """ Fixture to return categorical features."""
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_data_shape(data):
    """ If data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_process_data_shape_output(data, categorical_features):
    """ Test the correct shape of the data after the process data function"""
    x, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )
    assert x.shape[0] == data.shape[0], "Number of rows are different after aplying process_data in features dataset"
    assert y.shape[0] == data.shape[0], "Number of rows are different after aplying process_data in target dataset"


def test_process_data_datatype_output(data, categorical_features):
    """ Test the correct shape of the data after the process data function"""
    x, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )
    assert isinstance(x, np.ndarray), "Features dataset is not a numpy array"
    assert isinstance(y, np.ndarray), "Target dataset is not a numpy array"
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder), "Encoder is not a OneHotEncoder"
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer), "Lb is not a LabelBinarizer"
