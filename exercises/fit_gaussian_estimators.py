from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
# import plotly.graph_objects as go
# import plotly.io as pio
from matplotlib import pyplot as plt

# pio.templates.default = "simple_white"


def test_univariate_gaussian():



    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    univariategaussian = UnivariateGaussian().fit(np.random.normal(loc=mu, scale=sigma, size=1000))
    print("({},{})".format(univariategaussian.mu_, univariategaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mu = 10
    sigma = 1
    x = np.arange(10, 1000, 10)
    mu_array = []
    sigma_array = []
    for i in x:
        univariategaussian = UnivariateGaussian().fit(np.random.normal(loc=mu, scale=sigma, size=i))
        mu_array.append(np.abs(univariategaussian.mu_ - mu))
    plt.plot(x, mu_array)
    plt.legend(["mu"])
    plt.xlabel("Sample Size")
    plt.ylabel("Absolute distance from real value")
    plt.title("The different errors according to different sample size.")
    plt.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    univariategaussian = UnivariateGaussian().fit(np.random.normal(loc=mu, scale=sigma, size=1000))
    t = np.linspace(6, 14, 1000)
    y = univariategaussian.pdf(t)
    plt.scatter(t, y)
    plt.legend(["PDF"])
    plt.xlabel("Sample value")
    plt.ylabel("Density of probability")
    plt.title("PDF function for mu=10, sigma=1")  # TODO: answer the Q3
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0]).T
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)
    mvg = MultivariateGaussian().fit(samples)
    print()
    print("expectations:")
    print(mvg.mu_)
    print("cov matrix:")
    print(mvg.cov_)

    # Question 5 - Likelihood evaluation
    f2 = f1 = np.linspace(-10, 10, 200)
    y = np.zeros(shape=(200, 200))
    b = True
    max = 0
    maxi = (0, 0)
    samples = np.random.multivariate_normal(mu, cov, 1000)
    for i in range(200):
        for j in range(200):
            mu = np.array([f1[i], 0, f2[j], 0]).T
            y[i][j] = MultivariateGaussian.log_likelihood(mu, cov, samples)
            if b:
                max = y[i][j]
                b = False
            if max < y[i][j]:
                max = y[i][j]
                maxi = (i, j)

    plt.imshow(y)
    plt.show()

    # Question 6 - Maximum likelihood
    print(maxi)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
