import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.io as pio

HOUSE_PRICES_CSV = r"C:\Users\t8864522\Documents\GitHub\IML.HUJI\datasets\house_prices.csv"


# pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    return df


def clear_data(df: pd.DataFrame):
    df.dropna(inplace=True)
    df.drop(df.loc[df.zipcode == 0].index, inplace=True)
    df.drop(df.loc[df.price < 0].index, inplace=True)
    # df.drop(df.loc[df.bathrooms < 0].index, inplace=True)
    # df.drop(df.loc[df.zipcode == 0 or df.date < 0 or df.bathrooms < 0].index, inplace=True)
    df["yr_renovated"] = np.maximum(df["yr_renovated"], df["yr_built"])
    # for fet in ["date"]
    df.drop(["id", "date", "waterfront", "zipcode"], axis=1, inplace=True)
    return df.drop(labels="price", axis=1), df["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are sa ed
    """
    for col in X.columns:
        plt.scatter(df[col], response)
        pearson = np.cov(df[col], y) / (np.std(df[col]) * np.std(y))
        plt.title("for {} with the pearson correlation of {}".format(col, pearson[0, 1]))
        plt.savefig(r"{}\{}.png".format(output_path, col))
        plt.clf()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, response = clear_data(load_data(filename=HOUSE_PRICES_CSV))

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, response, "aaa")

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(df, response, train_proportion=.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    std_loss = np.zeros(len(list(range(10, 100))))
    t = np.zeros(len(list(range(10, 100))))
    y = np.zeros(len(list(range(10, 100))))
    for p in range(10, 100):
        loss = list()
        pre = p / 100
        for i in range(10):
            x_train1, y_train1, _, _ = split_train_test(x_train, y_train, train_proportion=pre)
            lin = LinearRegression()
            lin.fit(x_train1.to_numpy(), y_train1.to_numpy())
            loss.append(lin.loss(x_test.to_numpy(), y_test.to_numpy()))
        std_loss[p - 10] = 2 * np.std(loss)
        y[p - 10] = np.mean(loss)
        t[p - 10] = p
    # plt.scatter(t, y)
    plt.errorbar(t, y, std_loss)
    plt.xlabel("percentage of the train used")
    plt.ylabel("loss")
    plt.show()
