import os
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score

os.chdir(r'g:\\Programs\\python\\Machine Learning\\Datasets')

# LOAD DATASET
train_data_filename = "heartdisease-train.csv"
test_data_filename  = "heartdisease-test.csv"
training_data = pd.read_csv(train_data_filename)
test_data     = pd.read_csv(test_data_filename)

# PREPROCESSING DATA
training_data = MinMaxScaler().fit_transform(training_data)
test_data     = MinMaxScaler().fit_transform(test_data)

# TRAINING SET
# NUMPY NDARRAY
X_train = training_data[:, 0:12]
Y_train = training_data[:, 13]
# PANDAS DATAFRAME
# X_train = training_data.loc[:, training_data.columns[:-1]]
# Y_train = training_data.loc[:, training_data.columns[-1:]]

# TEST SET
X_test = test_data[:, 0:12]
Y_test = test_data[:, 13]
# PANDAS DATAFRAME
# X_test = test_data.loc[:, test_data.columns[:-1]]
# Y_test = test_data.loc[:, test_data.columns[-1:]]

# TRAINS AND CREATES A MODEL
linear_regression_model   = LinearRegression().fit(X_train, Y_train)
logistic_regression_model = LogisticRegression().fit(X_train, Y_train)

# PREDICTS
linear_predicted_y   = linear_regression_model.predict(X_test)
logistic_predicted_y = logistic_regression_model.predict(X_test)

# EVALUATES MODEL (REGRESSION, use r2_score)
linear_score   = r2_score(Y_test, linear_predicted_y)
logistic_score = r2_score(Y_test, logistic_predicted_y)

print("Linear Regression Accuracy: {}%".format(linear_score * 100))
print("Logistic Regression Accuracy: {}%".format(logistic_score * 100))

scores = cross_val_score(LogisticRegression(), X_train, Y_train, cv=10, scoring="neg_mean_squared_error")
print(scores)
print("Cross validation score: {}".format(scores.mean()))