from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
            exact_x = np.zeros((len(indexs), len(X[0])))
            for i in range(len(indexs)):
                exact_x[i] = X[indexs[i]]
            exact_x_tran = exact_x.transpose()
            for f in range(n_features):
                self.mu_[k][f] += np.average(exact_x_tran[f])
        self.vars_ = np.zeros(shape=(n_classes, n_features))
        for k in range(n_classes):
            for d in range(n_features):
                sumy = 0
                counter = 0
                for i in range(len(X)):
                    indic = 1 if y[i] == self.classes_[k] else 0
                    counter += indic
                    sumy += np.power(indic * (X[i, d] - self.mu_[k, d]), 2)
                self.vars_[k, d] = sumy / (counter - 1)
        self.pi_ = np.zeros(n_classes)
        for k in y:
            k_index = np.where(self.classes_ == k)[0]
            self.pi_[k_index] += 1
        for i in range(n_classes):
            self.pi_[i] = self.pi_[i] / len(X)

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

        l = self.likelihood(X)  # np.ndarray of shape (n_samples, n_classes)
        c = np.argmax(l, axis=1)
        map(lambda x: self.classes_[x], c)
        return c

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

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
        l = np.zeros((len(X), len(self.classes_)))
        for s in range(len(X)):
            for k in range(len(self.classes_)):
                value = 0
                for d in range(n_features):
                    value += -0.5 * np.log(2 * np.pi) - np.log(np.sqrt(self.vars_[k, d])) - \
                             (np.power(X[s, d] - self.mu_[k, d], 2) / (2 * self.vars_[k, d]))
                l[s, k] = value + np.log(self.pi_[k])
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
