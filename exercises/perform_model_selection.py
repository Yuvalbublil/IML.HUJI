from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X = np.linspace(-1.2, 2, n_samples)
    np.random.shuffle(X)
    # X = np.random.uniform(low=-1.2, high=2, size=n_samples)
    eps = np.random.normal(0, noise, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = f(X) + eps
    train_x, train_y, test_x, test_y = split_train_test(X, y, 2 / 3)
    plt.plot(np.sort(X), f(np.sort(X)), 'k')
    plt.scatter(train_x, train_y)
    plt.scatter(test_x, test_y)
    plt.legend(["real function", "train_data", "test_data"])
    plt.title(f"train and test data with the real function witout noise\n{n_samples} samples and noise="
              f" {noise}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(0, 11, 1)
    train_score = np.zeros(11)
    test_score = np.zeros(11)
    for d in range(11):
        train_score[d], test_score[d] = cross_validate(PolynomialFitting(d), X, y, mean_square_error, 5)
    plt.plot(degrees, train_score, 'o-')
    plt.plot(degrees, test_score, 'o-')
    plt.legend(["train score", "test score"])
    plt.title(f"average error on CV with different degrees of polynomial fit\n{n_samples} samples and noise= {noise}")
    plt.xlabel("degree")
    plt.ylabel("average error")
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(test_score)
    print("-" * 40)
    print(f"{n_samples} samples and noise= {noise}")
    print(f"the best k is {k_star}")
    esti = PolynomialFitting(k_star)
    esti.fit(train_x, train_y)
    print(f"loss is {esti.loss(test_x, test_y):.2f}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = split_train_test(x, y, n_samples / len(x))

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam = np.linspace(0.002, 2, n_evaluations)
    ridge_error_train = np.zeros(n_evaluations)
    ridge_error_test = np.zeros(n_evaluations)
    lasso_error_train = np.zeros(n_evaluations)
    lasso_error_test = np.zeros(n_evaluations)

    for i, l in enumerate(lam):
        ridge_error_train[i], ridge_error_test[i] = cross_validate(RidgeRegression(l), train_x, train_y,
                                                                   mean_square_error)
        lasso_error_train[i], lasso_error_test[i] = cross_validate(Lasso(l), train_x, train_y, mean_square_error)

    plt.plot(lam, ridge_error_train)
    plt.plot(lam, ridge_error_test)
    plt.plot(lam, lasso_error_train)
    plt.plot(lam, lasso_error_test)

    plt.legend(["ridge error train", "ridge error test", "lasso error train", "lasso error test"])
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(1500, 10)
    select_regularization_parameter()
