import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, MinMaxScaler
import pandas as pd

from sklearn.model_selection import train_test_split

import os

os.chdir(r'g:\\Programs\\python\\Machine Learning\\Linear Regression')

# LOAD data set
dataset = pd.read_csv('heartdisease-train.csv', header=None)
dataset.head()
# SCALE DATA
scaler = MinMaxScaler()
df = scaler.fit_transform(dataset[dataset.columns]) 

# print(df[:,3]) # FETCH COLUMN VALUES IN NUMPY ARRAY

# LOAD SCALED DATA
set_x = df[:, :-1] # CONTAINS ALL COLUMNS EXCEPT THE LAST 1
set_y = df[:, 1]
# set_x = set_x.reshape(-1, 1)
set_y = set_y.reshape(-1, 1)
# print(set_y)

# set_x = dataset[dataset.columns[0]]
# set_y = dataset[dataset.columns[3]]

x_train, x_test, y_train, y_test = train_test_split(set_x, set_y, test_size=0.2, random_state=0)
regression_model = linear_model.LinearRegression()
regression_model.fit(x_train, y_train)

# coeff_df = pd.DataFrame(regression_model.coef_, set_x.columns, columns='Coefficient')
print(regression_model.coef_)
print(dataset.describe())
# print(coeff_df)

#plt.plot(set_x, set_y, "o")
#plt.plot(set_x, regression_model.predict(set_x))
#plt.show()

# # Load the diabetes dataset
# diabetes = datasets.load_diabetes()

# print(type(diabetes))

# # Use only one feature
# diabetes_X = diabetes.data[:, np.newaxis, 2]

# print(diabetes_X)

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# # # Plot outputs
# # plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# # plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# # plt.xticks(())
# # plt.yticks(())

# # plt.show()