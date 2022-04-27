from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = len(X[0])
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.mu_ = np.zeros(shape=(n_classes, n_features))
        for k in range(len(self.classes_)):
            indexs = np.where(y == self.classes_[k])[0]
            exact_x = np.zeros(len(indexs))
            for i in range(len(indexs)):
                exact_x[i] = X[indexs[i]]
            exact_x_tran = exact_x.transpose()
            for f in range(n_features):
                self.mu_[k][f] += np.average(exact_x_tran[f])

        self.cov_ = np.zeros(shape=(n_features, n_features))
        mu_t = self.mu_.transpose()
        for i in range(len(X)):
            k = y[i]
            k_index = np.where(self.classes_ == k)[0]
            self.cov_ = self.cov_ + np.matmul(X[i] - mu_t[k_index], (X[i] - mu_t[k_index]).transpose())
        self.cov_ = np.divide(self.cov_, len(X) - n_classes)
        self._cov_inv = inv(self.cov_)
        self.pi_ = np.zeros(n_classes)
        for k in y:
            k_index = np.where(self.classes_ == k)[0]
            self.pi_[k_index] += 1
        for i in range(n_classes):
            self.pi_ = self.pi_ / len(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        y = np.zeros(len(X))
        for i in range(len(X)):
            max_index = 0
            max_value = None
            for k in range(len(self.classes_)):
                a: np.ndarray = np.matmul(self._cov_inv, self.mu_[k])
                b = np.log(self.pi_[k]) - 0.5 * np.matmul(np.matmul(self.mu_[k], self._cov_inv), self.mu_[k])
                value = np.matmul(a.transpose(), X[i]) + b
                if max_value is None or value > max_value:
                    max_value = value
                    max_index = k
            y[i] = self.classes_[max_index]
        return y

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data o ver the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        n_features = len(X[0])
        l = np.zeros(shape=(len(X), n_features))
        for i in range(len(X)):
            for j in range(len(self.classes_)):
                l[i, j] = np.log(self.pi_[j]) - 0.5 * n_features * np.log(2 * np.pi) - 0.5 * np.log(
                    det(self.cov_)) - 0.5 * np.matmul(np.matmul((X[i]-self.mu_[j]).transpose(), self._cov_inv), X[i]-self.mu_[j])
        return l

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
