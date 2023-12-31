import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, kernel_approximation, metrics, svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.svm import SVC


def _loading_example_dataset() -> None:
    """
    This function covers the first part of the tutorial that
    shows how data is loaded and analysed
    """

    digits = datasets.load_digits()

    print("\n")
    print(digits.data)
    print("\n")

    print(f"Target: {digits.target}")
    print(f"Shape: {digits.data.shape}")
    print("\n")

    # The images array is constrained by the row dimension
    # found from digits.data.shape.   It looks like
    # digits.target are the numbers of the digit images we
    # try to learn.   Not sure why 8 appear twice yet.
    print(f"Image[0]: \n{digits.images[0]}\n")

    print(f"Image[1]: \n{digits.images[1]}\n")

    print(f"Image[1796]: \n{digits.images[1796]}\n")


def _recognizing_hand_written_digits() -> None:
    """
    This is a sub section/example within the previous tutorial

    The digits dataset consists of 8x8 pixel images of digits.
    The images attribute of the dataset stores 8x8 arrays of grayscale
    values for each image. We will use these arrays to visualize the
    first 4 images. The target attribute of the dataset stores the
    digit each image represents and this is included in the title of
    the 4 plots below.
    """
    digits = datasets.load_digits()

    _, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 3))

    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    plt.show()

    """
    To apply a classifier on this data, we need to flatten the images,
    turning each 2-D array of grayscale values from shape (8, 8) into
    shape (64,). Subsequently, the entire dataset will be of shape
    (n_samples, n_features), where n_samples is the number of images
    and n_features is the total number of pixels in each image.

    We can then split the data into train and test subsets and fit
    a support vector classifier on the train samples.
    """

    # flatten the images
    n_samples = len(digits.images)
    print(f"n_samples: {n_samples}")
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    plt.show()

    # Generate a classification report
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Genrate confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

    """
    If the results from evaluating a classifier are stored in the form of
    a confusion matrix and not in terms of y_true and y_pred, one can still
    build a classification_report as follows:
    """

    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground
    # truths and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )


def _conventions() -> None:
    """
    Type Casting:
    Where possible, input of type float32 will maintain its data
    type. Otherwise input will be cast to float64.
    """

    rng = np.random.RandomState(0)
    X = rng.rand(10, 2000)
    X = np.array(X, dtype="float32")
    print(X.dtype)

    transformer = kernel_approximation.RBFSampler()
    X_new = transformer.fit_transform(X)
    print(X_new.dtype)

    # Here, the first predict() returns an integer array,
    # since iris.target (an integer array) was used in fit.
    # The second predict() returns a string array, since
    # iris.target_names was for fitting.

    iris = datasets.load_iris()
    clf = SVC()
    print(clf.fit(iris.data, iris.target))
    # SVC()

    print(list(clf.predict(iris.data[:3])))
    # [0, 0, 0]

    print(clf.fit(iris.data, iris.target_names[iris.target]))
    # SVC()

    print(list(clf.predict(iris.data[:3])))
    # ['setosa', 'setosa', 'setosa']

    # Refitting and updating parameters

    #   Hyper-parameters of an estimator can be updated after it
    #   has been constructed via the set_params() method. Calling
    #   fit() more than once will overwrite what was learned by
    #   any previous fit():

    X, y = load_iris(return_X_y=True)

    clf = SVC()
    print(clf.set_params(kernel="linear").fit(X, y))
    # SVC(kernel='linear')

    print(clf.predict(X[:5]))
    # array([0, 0, 0, 0, 0])

    print(clf.set_params(kernel="rbf").fit(X, y))
    # SVC()

    print(clf.predict(X[:5]))
    # array([0, 0, 0, 0, 0])

    # MultiClass vs MultiLabel fitting

    # In the above case, the classifier is fit on a 1d array
    # of multiclass labels and the predict() method therefore
    # provides corresponding multiclass predictions
    X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
    y = [0, 0, 1, 1, 2]
    classif = OneVsRestClassifier(estimator=SVC(random_state=0))
    print(classif.fit(X, y).predict(X))
    # array([0, 0, 1, 1, 2])

    # Here, the classifier is fit() on a 2d binary label
    # representation of y, using the LabelBinarizer. In this
    # case predict() returns a 2d array representing the
    # corresponding multilabel predictions.

    y = LabelBinarizer().fit_transform(y)
    print(classif.fit(X, y).predict(X))
    # array([[1, 0, 0],
    #        [1, 0, 0],
    #        [0, 1, 0],
    #        [0, 0, 0],
    #        [0, 0, 0]])

    # In this case, the classifier is fit upon instances
    # each assigned multiple labels. The MultiLabelBinarizer
    # is used to binarize the 2d array of multilabels to fit
    # upon. As a result, predict() returns a 2d array with
    # multiple predicted labels for each instance.

    y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
    y = MultiLabelBinarizer().fit_transform(y)
    print(classif.fit(X, y).predict(X))
    # array([[1, 1, 0, 0, 0],
    #        [1, 0, 1, 0, 0],
    #        [0, 1, 0, 1, 0],
    #        [1, 0, 1, 0, 0],
    #        [1, 0, 1, 0, 0]])


if __name__ == "__main__":

    _loading_example_dataset()

    _recognizing_hand_written_digits()

    _conventions()
