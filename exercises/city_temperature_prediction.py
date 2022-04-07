import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
# import plotly.express as px
# import plotly.io as pio
# pio.templates.default = "simple_white"
import matplotlib.pyplot as plt


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df.dropna(inplace=True)
    df.drop(df.loc[df.Temp < -40].index, inplace=True)
    df["day_of_year"] = df["Date"].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\Users\t8864522\Documents\GitHub\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel: pd.DataFrame = df.loc[df["Country"] == "Israel"]
    # print(df_israel["Year"].unique())
    legend = []
    for year in df_israel["Year"].unique():
        df_temp: pd.DataFrame = df_israel.loc[df_israel["Year"] == year]
        plt.scatter(df_temp["day_of_year"], df_temp["Temp"])
        legend.append(str(year))
    plt.legend(legend)
    plt.show()
    groups = df_israel[["Month", "Temp"]].groupby("Month")
    agg = groups.agg(["std"])
    plt.bar(agg.index, agg[('Temp', 'std')])
    plt.xticks(np.arange(1, 13))
    plt.xlabel("Month")
    plt.ylabel("STD")
    plt.show()

    # Question 3 - Exploring differences between countries
    groups = df[["Month", "Temp", "Country"]].groupby("Country")
    legend = []
    for country, inside_df in groups:
        legend.append(country)
        df1 = inside_df[["Month", "Temp"]].groupby("Month")
        agg = df1.agg(["std", "mean"])
        plt.errorbar(agg.index, agg[('Temp', 'mean')], agg[('Temp', 'std')])
        plt.xticks(np.arange(1, 13))
        plt.xlabel("Month")
        plt.ylabel("mean with std bar")
    plt.legend(legend)
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    x_train, y_train, x_test, y_test = split_train_test(df_israel.drop(["Temp"], axis=1)["day_of_year"],
                                                        df_israel["Temp"],
                                                        train_proportion=.75)
    loss = []
    for k in range(1, 11):
        polynomialfitter = PolynomialFitting(k)
        polynomialfitter.fit(x_train.to_numpy(), y_train.to_numpy())
        loss.append(np.round(polynomialfitter.loss(x_test.to_numpy(), y_test.to_numpy()), 2))
        print("{}: loss = {}".format(k, loss[-1]))
    plt.bar(list(range(1, 11)), loss)
    plt.xticks(np.arange(1, 11))
    plt.xlabel("k")
    plt.ylabel("loss")
    plt.title("the loss as a function of k.")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    x_train, y_train = df_israel.drop(["Temp"], axis=1)["day_of_year"], df_israel["Temp"]
    polynomialfitter = PolynomialFitting(3)
    polynomialfitter.fit(x_train.to_numpy(), y_train.to_numpy())

    groups = df[["day_of_year", "Temp", "Country"]].groupby("Country")
    legend = []
    loss = []
    for country, inside_df in groups:
        if country == "Israel":
            continue
        legend.append(country)
        loss.append(polynomialfitter.loss(inside_df["day_of_year"], inside_df["Temp"]))
    plt.bar(legend, loss)
    plt.ylabel("loss")
    plt.title("loss for each county fitted on Israel")
    plt.show()