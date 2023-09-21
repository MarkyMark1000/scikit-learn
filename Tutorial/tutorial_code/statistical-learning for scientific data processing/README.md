# NOTES

Supervised learning consists in learning the link between
two datasets: the observed data X and an external variable
y that we are trying to predict, usually called “target” or
“labels”. Most often, y is a 1D array of length n_samples.

All supervised estimators in scikit-learn implement a fit(X, y)
method to fit the model and a predict(X) method that,
given unlabeled observations X, returns the predicted labels y.

classification: classify observations to a set of finite labels,
                in other words to “name” the objects observed

regression: goal is to predict a continuous target variable
