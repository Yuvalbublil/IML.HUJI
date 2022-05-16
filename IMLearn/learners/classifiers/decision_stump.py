from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import weighted_misclassification_error
from itertools import product

EPSILON = 1.0


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        x_tran = X.transpose()
        err = np.zeros((n_features, 2))
        thr = np.zeros((n_features, 2))
        for f in x_tran:
            thr[f, 0], err[f, 0] = self._find_threshold(f, y, -1)
            thr[f, 1], err[f, 1] = self._find_threshold(f, y, 1)
        self.j_, self.sign_ = np.unravel_index(np.argmin(err, axis=None), err.shape)
        self.threshold_ = thr[self.j_, (1 + self.sign_)/2]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        x_tran = X.transpose()
        x_j = x_tran[self.j_]

        def apply(x):
            return self.sign_ if x >= self.threshold_ else -self.sign_

        return np.array(list(map(apply, x_j)))

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        def mini_pred(X, thr):
            x_tran = X.transpose()
            x_j = x_tran

            def apply(x):
                return sign if x >= thr else -sign

            return np.array(list(map(apply, x_j)))

        test_values = np.hstack((values, np.array([values[-1].item() + EPSILON], dtype=object)))
        thr_err = np.zeros(len(test_values))
        for i in range(len(test_values)):
            thr_err[i] = weighted_misclassification_error(labels, mini_pred(values, test_values[i]))
        min_ind = np.argmin(thr_err)
        return test_values[min_ind], thr_err[min_ind]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return weighted_misclassification_error(y, self.predict(X))
