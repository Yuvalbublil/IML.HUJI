import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics import weighted_misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models = list()
        n_samples = len(X)
        self.weights_ = np.zeros(self.iterations_)
        self.D_ = np.zeros((self.iterations_, n_samples))
        D = np.ones(n_samples) / n_samples
        for t in range(self.iterations_):
            self.D_[t] = D
            self.models.append(self.wl_())
            self.models[-1].fit(X, y * D)
            epsilon = weighted_misclassification_error(y*D, self.models[-1].predict(X))
            w = 0.5 * np.log(1 / epsilon - 1)
            self.weights_[t] = w
            D = D * np.exp(-y * w * self.models[t].predict(X))
            sum_D = np.sum(D)
            D = D / sum_D

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_:
            raise ValueError("T can't be greater the iterations")
        n_samples = len(X)
        pred = np.ndarray((T, n_samples))
        for t in range(1, T+1):
            pred[t-1] = self.models[t-1].predict(X)
        pred_t = pred.T
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = np.sign(np.sum(pred_t[i] * self.weights_[:T]))
        return y_pred

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return weighted_misclassification_error(y, self.partial_predict(X, T))
