import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r'g:\\Programs\\python\\Machine Learning\\Datasets')

datas = pd.read_csv('advertising_revenue_example.csv')
print(datas.shape)
datas.head()

X = datas[datas.columns[0]].values
Y = datas[datas.columns[1]].values
print(X)
print(Y)

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

## SEARCH FOR M & (C)OEF
m_atas = 0
m_bawah = 0
for i in range(n):
    m_atas += (X[i] - mean_x) * (Y[i] - mean_y)
    m_bawah +=  (X[i] - mean_x)**2

# mean_y = m * mean_x + c
m = m_atas / m_bawah
c = mean_y - (m * mean_x)
print("{} = {} * {} + {}".format(mean_y, m, mean_x, c))
print(mean_y)
predicted_y = []

for i in range (len(X)):
    y = m * X[i] + c
    predicted_y.append(y) # h_theta_xi

ss_t = 0
ss_r = 0

for i in range (n):
    y_predic = m * X[i] + c
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - predicted_y[i]) ** 2
Error = ss_r / ss_t
r2 = 1 - Error
print(r2)

# plt.axis([0, 100, 0, 100])
plt.plot(X, Y, "o")
plt.plot(X, predicted_y)
plt.show()
