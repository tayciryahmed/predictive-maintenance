import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

import numpy as np
from sklearn.metrics import mean_absolute_error


class RelativeRMSE(rw.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) /(y_true+1))))


problem_title = 'Predictive maintenance'
_target_column_name = 'RUL'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

score_types = [
    RelativeRMSE(name='Relative RMSE', precision=3),
    #rw.score_types.RMSE(name='rmse', precision=1),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.5, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=" ")
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.txt'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.txt'
    return _read_data(path, f_name)
