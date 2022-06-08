from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

def combine (X):
    y = X[0]
    for i,x in enumerate(X):
        if i != 0:
            y = np.append(y, x, axis=0)
    return y

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    arrays_X = np.array_split(X, cv)
    arrays_y = np.array_split(y, cv)
    test_score = 0
    train_score = 0
    for i in range(cv):
        train_x = combine(np.delete(arrays_X, i, axis=0))
        train_y = combine(np.delete(arrays_y, i, axis=0))
        test_x = arrays_X[i]
        test_y = arrays_y[i]
        estimator.fit(train_x, train_y)
        train_score += scoring(train_y, estimator.predict(train_x))
        test_score += scoring(test_y, estimator.predict(test_x))
    return train_score / cv, test_score / cv
