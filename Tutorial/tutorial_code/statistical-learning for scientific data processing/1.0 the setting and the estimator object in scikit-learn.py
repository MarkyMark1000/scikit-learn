import matplotlib.pyplot as plt
from sklearn import datasets


def _datasets() -> None:
    """
    Learning information from one or more datasets that are represented
    as 2D arrays.

    First axis of these arrays is the samples axis, while the second is
    the features axis
    """

    iris = datasets.load_iris()
    data = iris.data
    print(f"Iris Data Shape:\n{data.shape}\n")
    print(f"Iris Features:\n{iris.DESCR}\n")

    # When the data is not initially in the (n_samples, n_features) shape,
    # it needs to be preprocessed in order to be used by scikit-learn.

    # An example of reshaping data (seen before - digits)

    digits = datasets.load_digits()
    print(f"digits shape:\n{digits.images.shape}\n")
    # (1797, 8, 8)

    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
    plt.savefig("./Images/output_plot.png")

    # To use the data, we reshape it:
    data = digits.images.reshape((digits.images.shape[0], -1))
    print(f"New data shape:\n{data.shape}\n")


def _estimator() -> None:
    """
    This is just for some notes on the tutorial:

    Estimators objects:

    An estimator is any object that learns from data; it may be a
    classification, regression or clustering algorithm or a transformer
    that extracts/filters useful features from raw data.   All estimator
    objects expose a fit method that takes a dataset (usually a 2-d array):

        estimator.fit(data)

    Estimator parameters: All the parameters of an estimator can be set
    when it is instantiated or by modifying the corresponding attribute:

        estimator = Estimator(param1=1, param2=2)
        estimator.param1

    Estimated parameters: When data is fitted with an estimator, parameters
    are estimated from the data at hand. All the estimated parameters are
    attributes of the estimator object ending by an underscore:

        estimator.estimated_param_
    """

    pass


if __name__ == "__main__":

    _datasets()

    _estimator()
