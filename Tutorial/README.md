# OVERVIEW

This folder is going to be used for the scikit-learn version 1.3.0 tutorial,
taken from the following location:

https://scikit-learn.org/stable/tutorial/index.html


## Tutorial Notes
---

Machine learning is generally split into two large categories:

* supervised learning

* unsupervised learning

Training and Testing Set - it is common to split data into two sets, one to
train the algorithm on and a second used to test the trained algorithm on.

### Confusion Matrix
---

This is a table used to help assess where errors in the model were made.

The rows represent the actual classes the outcomes should have been.
While the columns represent the predictions we have made. Using this table it
is easy to see which predictions are wrong.

### Classification Report
---

This is helpful for displaying metrics on how accurate the model was:

Accuracy: measures how often the model is correct.

Precision: Of the positives predicted, what percentage is truly positive?

Sensitivity or Recall: Of all the positive cases, what percentage are predicted positive?

F Score: F-score is the "harmonic mean" of precision and sensitivity.

Support: Number of actual occurances in the dataset.

There is a basic example here, which may be of use:
https://www.w3schools.com/python/python_ml_confusion_matrix.asp

What are good values for these statistics (from google):

            f1-score    Accuracy    Recall
Excellent   > 0.85                  > 0.8
Good        > 0.7        > 0.7      > 0.7

### Loading from external datasets
---

There is a section, with little code here:
https://scikit-learn.org/stable/datasets/loading_other_datasets.html#external-datasets

### Learning and Predicting
---

The two key routines are fit() and predict(), which fit the model to training data
and then try to predict results on test data.

The tutorial uses a support vector machine model with arguments of gamma=0.001.

This value was input manually, but can be estimated using grid search.

```
Two generic approaches to parameter search are provided in scikit-learn: for given
values, GridSearchCV exhaustively considers all parameter combinations, while
RandomizedSearchCV can sample a given number of candidates from a parameter space
with a specified distribution. Both these tools have successive halving counterparts
HalvingGridSearchCV and HalvingRandomSearchCV, which can be much faster at finding
a good parameter combination.
```

https://scikit-learn.org/stable/modules/grid_search.html#grid-search



## Outstanding Questions
---
