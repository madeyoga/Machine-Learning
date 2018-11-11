
from Models.Classification.ClassifierModel import *

from DataManager import Manager

import numpy as np

import pandas as pd
import os

abspath             = os.path.abspath(__file__)
this_script_path    = os.path.dirname(abspath)
datasets_path       = this_script_path + "\\Datasets"

os.chdir(datasets_path)

training_data_name  = "heartdisease-train.csv" # "iris-train.csv" # input("input training dataset file name: ")
test_data_name      = "heartdisease-test.csv"  # "iris-test.csv"  # input("input test dataset file name : ")

training_dataset    = pd.read_csv(training_data_name)
test_dataset        = pd.read_csv(test_data_name)

manager = Manager(training_dataset=training_dataset, test_dataset=test_dataset)
manager.scale_data()

print(manager.get_compare_frame())

# print(manager.KNN_model)

# knn_pred = manager.get_KNN_prediction()
# pd_knn = pd.DataFrame(
#     data={'knn predictions' : knn_pred}
# )
# pd_compare = manager.Y_test.join(pd_knn)

# gnb_pred = manager.get_GNB_prediction()
# pd_gnb = pd.DataFrame(
#     data={'gnb predictions' : gnb_pred}
# )
# pd_compare = pd_compare.join(pd_gnb)
# print(pd_compare)

# print(manager.get_KNN_accuracy())
# print(manager.get_GNB_accuracy())