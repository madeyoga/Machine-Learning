from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd

os.chdir(r'g:\\Programs\\python\\Machine Learning\\Bayes')

iris = datasets.load_iris()

data = pd.read_csv("heartdisease-train.csv", header=None)
test_set = pd.read_csv("heartdisease-test.csv", header=None)
# X = np.array(data.loc[:, data.columns[:-1]])
# Y = np.array(data.loc[:, data.columns[-1:]])
X = data.loc[:, data.columns[:-1]]
Y = data.loc[:, data.columns[-1:]]

# PREPROCESSING DATA
scaled_X = MinMaxScaler().fit_transform(X)
# scaled_Y = MinMaxScaler().fit_transform(Y)

# MODELS
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, Y.values.ravel())
y_pred = knn.predict(test_set.loc[:, test_set.columns[:-1]])
score0 = accuracy_score(test_set.loc[:, test_set.columns[-1:]], y_pred)
print("Prediction\n{}".format(y_pred))
print("KNN Accuracy: {}%".format(score0 * 100))

knn.fit(scaled_X, Y.values.ravel())
y_pred = knn.predict(test_set.loc[:, test_set.columns[:-1]])
score0 = accuracy_score(test_set.loc[:, test_set.columns[-1:]], y_pred)
print("Prediction\n{}".format(y_pred))
print("KNN Accuracy(Scaled Value): {}%".format(score0 * 100))