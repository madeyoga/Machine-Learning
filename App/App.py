import time
from Models.Classification.ClassifierModel import *
from DataManager import Manager

import numpy as np

import pandas as pd
import os

abspath             = os.path.abspath(__file__)
this_script_path    = os.path.dirname(abspath)
datasets_path       = this_script_path + "\\Datasets"

os.chdir(datasets_path)

training_data_filename  = "iris-train.csv"
test_data_filename      = "iris-test.csv"

# training_data_filename  = "heartdisease-train.csv"
# test_data_filename      = "heartdisease-test.csv"  

# training_data_filename  = input("input training dataset file name: ")
# test_data_filename      = input("input test dataset file name : ")

training_dataset    = pd.read_csv(training_data_filename)
test_dataset        = pd.read_csv(test_data_filename)

print("Load Dataset... ({}/{})".format(training_data_filename, test_data_filename))
time.sleep(1)
manager = Manager(training_dataset=training_dataset, test_dataset=test_dataset)
manager.scale_data()

## INITIAL KNN MODEL
KNN_model = KNeighborsModel(n=5)
KNN_model.train(X=manager.X_train, Y=manager.Y_train)
KNN_model.predict_y(test_set_x=manager.X_test)

## INITIAL GNB MODEL
GNB_model = NBGaussModel()
GNB_model.train(X=manager.X_train, Y=manager.Y_train)
## PREDICT
GNB_model.predict_y(test_set_x=manager.X_test)

print("Test Accuracy: ")
print("KNN : {}%".format(KNN_model.get_accuracy(Y_test=manager.Y_test) * 100))
print("GNB : {}%".format(GNB_model.get_accuracy(Y_test=manager.Y_test) * 100))

KNN_model.predict_y(test_set_x=manager.X_train)
GNB_model.predict_y(test_set_x=manager.X_train)
print("Training Accuracy with training dataset: ")
print("KNN : {}%".format(KNN_model.get_accuracy(Y_test=manager.Y_train) * 100))
print("GNB : {}%".format(GNB_model.get_accuracy(Y_test=manager.Y_train) * 100))

manager.do_train_test_split()
GNB_model.train(X=manager.X_train, Y=manager.Y_train)
GNB_model.predict_y(test_set_x=manager.X_train)

knn_scores = []
k_n_range = range(1, 26)
for k in k_n_range:
   KNN_model.set_k(n=k)
   KNN_model.train(X=manager.X_train, Y=manager.Y_train)
   KNN_model.predict_y(test_set_x=manager.X_test)
   knn_scores.append(KNN_model.get_accuracy(Y_test=manager.Y_test))

print("Training Accuracy with train_test_split approach: ")
print("KNN : {}%".format(max(knn_scores) * 100))
print("GNB : {}%".format(GNB_model.get_accuracy(Y_test=manager.Y_train) * 100))

from matplotlib import pyplot as plt

plt.plot(k_n_range, knn_scores)
plt.xlabel("K")
plt.ylabel("Score")
plt.show()




