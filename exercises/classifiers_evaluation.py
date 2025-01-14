import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
from os import path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(path.join(r"C:\Users\t8864522\Documents\GitHub\IML.HUJI\datasets\\", f))
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def in_callable(p1: Perceptron, x1: np.ndarray, y1: int) -> None:
            losses.append(p1._loss(X, y))

        p = Perceptron(callback=in_callable)
        p.fit(X, y)
        # Plot figure of loss as function of fitting iteration
        plt.plot(np.arange(1, len(losses) + 1, 1), losses)
        plt.title(f"the loss over iterations on the {n} dataset.\n with {len(losses)} iterations")
        plt.ylabel("Loss")
        plt.xlabel("number of iterations")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(path.join(r"C:\Users\t8864522\Documents\GitHub\IML.HUJI\datasets\\", f))

        # Fit models and predict over training set
        lda = LDA()
        bayes = GaussianNaiveBayes()
        lda.fit(X, y)
        bayes.fit(X, y)
        y_pred_lda = lda.predict(X)
        y_pred_b = bayes.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(1, 2, subplot_titles=( f"Bayes with "
                                                                                                        f"accuracy of "
                                                                                           f""
                                                                                                         f""
                                                   f"{accuracy(y, y_pred_b):.5f}", f"LDA with accuracy of {accuracy(y, y_pred_lda):.5f}"))
        fig.update_layout(showlegend=False, title_text=f"analyzing the data from {f}")
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y_pred_lda,
                                                                                         symbol=y)), 1, 2)


        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y_pred_b,
                                                                                         symbol=y)), 1, 1)

        #
        # # Add traces for data-points setting symbols and colors

        #
        # # Add `X` dots specifying fitted Gaussians' means
        for col in [2, 1]:
            for center in range(len(lda.mu_)):
                fig.add_trace(go.Scatter(x=[lda.mu_[center][0]], y=[lda.mu_[center][1]], mode='markers',
                                         marker_color="black",
                                         marker_symbol=4, marker_size=10), col=col, row=1)
        #
        # # Add ellipses depicting the covariances of the fitted Gaussians
        for col, mu, cov in [(2, lda.mu_, lda.cov_), (1, bayes.mu_, bayes.vars_)]:
            var = cov
            for center in range(len(lda.mu_)):
                if col == 1:
                    cov = np.diag(var[center])
                fig.add_trace(get_ellipse(mu[center], cov), col=col, row=1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
