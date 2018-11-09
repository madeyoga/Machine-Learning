
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd

os.chdir(r'g:\\Programs\\python\\Machine Learning\\Datasets')

data = pd.read_csv("iris-train.csv", header=None)
test_set = pd.read_csv("iris-test.csv", header=None)
# X = np.array(data.loc[:, data.columns[:-1]])
# Y = np.array(data.loc[:, data.columns[-1:]])
X = data.loc[:, data.columns[:-1]]
Y = data.loc[:, data.columns[-1:]]

# PREPROCESSING DATA
scaled_X = MinMaxScaler().fit_transform(X)
# scaled_Y = MinMaxScaler().fit_transform(Y)

X_test = test_set.loc[:, test_set.columns[:-1]]
scaled_X_test = MinMaxScaler().fit_transform(X_test)

# MODELS
knn = KNeighborsClassifier(n_neighbors=5)

print(X_test, scaled_X_test)

knn.fit(X, Y.values.ravel())
y_pred = knn.predict(scaled_X_test)
score0 = accuracy_score(test_set.loc[:, test_set.columns[-1:]], y_pred)
print("Prediction\n{}".format(y_pred))
print("KNN Accuracy: {}%".format(score0 * 100))

knn.fit(scaled_X, Y.values.ravel())
y_pred = knn.predict(scaled_X_test)
score0 = accuracy_score(test_set.loc[:, test_set.columns[-1:]], y_pred)
print("Prediction\n{}".format(y_pred))
print("KNN Accuracy(Scaled Value): {}%".format(score0 * 100))