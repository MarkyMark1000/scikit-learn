import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier


def nearest_neighbour_curse_of_dimensionality() -> None:
    """
    Classification with 3 different types of irises
    (Setosa, Versicolour, and Virginica) from their
    petal and sepal length and width
    """

    iris_X, iris_y = datasets.load_iris(return_X_y=True)
    print(np.unique(iris_y))
    # array([0, 1, 2])

    # KNN Example (K Nearest Neighbours)

    # Split iris data in train and test data
    # A random permutation, to split the data randomly
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]

    # Create and fit a nearest-neighbor classifier
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)

    # KNeighborsClassifier()
    print(knn.predict(iris_X_test))
    # array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])

    print(iris_y_test)
    # array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])

    """
    For an estimator to be effective, you need the distance
    between neighboring points to be less than some value.
    In one dimension, this requires on average 1/d points.

    If the number of features is p, you now require 1/(d^p)
    points. As p becomes large, the number of training points
    required for a good estimator grows exponentially.

    This is called the curse of dimensionality and is a core
    problem that machine learning addresses.
    """


def linear_model_regression_to_sparsity() -> None:
    """
    The diabetes dataset consists of 10 physiological variables
    (age, sex, weight, blood pressure) measured on 442 patients,
    and an indication of disease progression after one year:
    The task at hand is to predict disease progression from
    physiological variables.
    """

    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Linear Regression

    regr = linear_model.LinearRegression()
    print(regr.fit(diabetes_X_train, diabetes_y_train))
    # LinearRegression()

    print(regr.coef_)
    # [0.30349955 -237.63931533  510.53060544  327.73698041 -814.13170937
    #   492.81458798  102.84845219  184.60648906  743.51961675   76.09517222]

    # The mean square error
    print(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    # 2004.5...

    # Explained variance score: 1 is perfect prediction
    # and 0 means that there is no linear relationship
    # between X and y.
    print(regr.score(diabetes_X_test, diabetes_y_test))
    # 0.585...


def shrinkage() -> None:
    """
    If there are few data points per dimension, noise in
    the observations induces high variance:
    A solution in high-dimensional statistical learning is
    to shrink the regression coefficients to zero: any two
    randomly chosen set of observations are likely to be
    uncorrelated. This is called Ridge regression:
    """

    X = np.c_[0.5, 1].T
    y = [0.5, 1]
    test = np.c_[0, 2].T
    regr = linear_model.LinearRegression()

    plt.figure()

    np.random.seed(0)
    for _ in range(6):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        plt.plot(test, regr.predict(test))
        plt.scatter(this_X, y, s=3)
    # LinearRegression...

    plt.savefig("./Images/2_output_plot.png")

    regr = linear_model.Ridge(alpha=0.1)
    plt.figure()

    np.random.seed(0)
    for _ in range(6):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        plt.plot(test, regr.predict(test))
        plt.scatter(this_X, y, s=3)

    plt.savefig("./Images/2B_output_plot.png")

    # This is an example of bias/variance tradeoff: the larger
    # the ridge alpha parameter, the higher the bias and the
    # lower the variance.

    # We can choose alpha to minimize left out error, this time
    # using the diabetes dataset rather than our synthetic data:

    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    alphas = np.logspace(-4, -1, 6)
    print(
        [
            regr.set_params(alpha=alpha)
            .fit(diabetes_X_train, diabetes_y_train)
            .score(diabetes_X_test, diabetes_y_test)
            for alpha in alphas
        ]
    )
    # [0.585..., 0.585..., 0.5854..., 0.5855..., 0.583..., 0.570...]


if __name__ == "__main__":

    nearest_neighbour_curse_of_dimensionality()

    linear_model_regression_to_sparsity()

    shrinkage()
