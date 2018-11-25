"""
* Neural Network model sangat sensitive pada data yg valuenya tidak di scale.
* Cross Validation score, menggunakan rata-rata
*
*
*
*
"""
from Models.Classification.ClassifierModel import KNeighborsModel
from Models.Classification.ClassifierModel import NBGaussModel
from Models.Classification.ClassifierModel import DTreeModel
from Models.Classification.ClassifierModel import MultiLayerPerceptronModel
from DataManager import Manager
from sklearn.model_selection import cross_val_score, KFold
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
manager = Manager(training_dataset=training_dataset, test_dataset=test_dataset)
# manager.scale_data()

## INITIAL KNN MODEL
KNN_model = KNeighborsModel(n=5)
KNN_model.train(X=manager.X_train, Y=manager.Y_train)
## INITIAL GNB MODEL
GNB_model = NBGaussModel()
GNB_model.train(X=manager.X_train, Y=manager.Y_train)
## INITIAL DTREE MODEL
DT_model = DTreeModel()
DT_model.train(X=manager.X_train, Y=manager.Y_train)
## INITIAL MULTI-LAYER PERCEPTRON MODEL
MLP_model = MultiLayerPerceptronModel()
MLP_model.train(X=manager.X_train, Y=manager.Y_train)
## PREDICT/TESTING
GNB_model.predict_y(test_set_x=manager.X_test)
KNN_model.predict_y(test_set_x=manager.X_test)
DT_model. predict_y(test_set_x=manager.X_test)
MLP_model.predict_y(test_set_x=manager.X_test)

scores_KNN = []
scores_GNB = []
scores_DT  = []
scores_MLP = []

print("Test Accuracy: ")
print("KNN : {}%".format(KNN_model.get_accuracy(Y_test=manager.Y_test) * 100))
print("GNB : {}%".format(GNB_model.get_accuracy(Y_test=manager.Y_test) * 100))
print("DT  : {}%".format(DT_model .get_accuracy(Y_test=manager.Y_test) * 100))
print("MLP : {}%".format(MLP_model.get_accuracy(Y_test=manager.Y_test) * 100))

scores_KNN.append(KNN_model.get_accuracy(Y_test=manager.Y_test) * 100)
scores_GNB.append(GNB_model.get_accuracy(Y_test=manager.Y_test) * 100)
scores_DT .append(DT_model .get_accuracy(Y_test=manager.Y_test) * 100)
scores_MLP.append(MLP_model.get_accuracy(Y_test=manager.Y_test) * 100)

#################################
### TRAIN TEST SPLIT APPROACH ###
#################################
manager.do_train_test_split()

## GNB
GNB_model.train(X=manager.X_train, Y=manager.Y_train)
GNB_model.predict_y(test_set_x=manager.X_test)
## KNN
knn_scores = []
k_n_range = range(1, 26)
for k in k_n_range:
   KNN_model.set_k(n=k)
   KNN_model.train(X=manager.X_train, Y=manager.Y_train)
   KNN_model.predict_y(test_set_x=manager.X_test)
   knn_scores.append(KNN_model.get_accuracy(Y_test=manager.Y_test))
## DT
DT_model.train(X=manager.X_train, Y=manager.Y_train)
DT_model.predict_y(test_set_x=manager.X_test)
## MLP
MLP_model.train(X=manager.X_train, Y=manager.Y_train)
MLP_model.predict_y(test_set_x=manager.X_test)

scores_KNN.append(max(knn_scores) * 100)
scores_GNB.append(GNB_model.get_accuracy(Y_test=manager.Y_train) * 100)
scores_DT .append(DT_model .current_accuracy * 100)
scores_MLP.append(MLP_model.current_accuracy * 100)

print("Training Accuracy with train_test_split approach: ")
print("KNN : {}%".format(int(max(knn_scores) * 100)))
print("GNB : {}%".format(int(GNB_model.get_accuracy(Y_test=manager.Y_train) * 100)))

#################################
### CROSS VALIDATION APPROACH ###
#################################
manager = Manager(training_dataset=training_dataset, test_dataset=test_dataset)
manager.scale_data()

## cross val StartifiedKFold
## KFold
# kf = KFold(n_splits=10)
print(cross_val_score(KNN_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
print(cross_val_score(GNB_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
print(cross_val_score(DT_model .model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
print(cross_val_score(MLP_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
scores_KNN.append(cross_val_score(KNN_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
scores_GNB.append(cross_val_score(GNB_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
scores_DT .append(cross_val_score(DT_model.model , manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)
scores_MLP.append(cross_val_score(MLP_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100)

from matplotlib import pyplot as plt

n_groups = 3

ax = plt.subplot()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

print(tuple(scores_KNN))

rects1 = ax.bar(
   index, tuple(scores_KNN), bar_width,
   alpha=opacity,
   color='b',
   label='KNN'
)

rects2 = ax.bar(
   index + bar_width, tuple(scores_GNB), bar_width,
   alpha=opacity,
   color='g',
   label='GNB'
)

rects3 = ax.bar(
   index + bar_width * 2, tuple(scores_DT), bar_width,
   alpha=opacity,
   color='y',
   label='DT'
)

rects4 = ax.bar(
   index + bar_width * 3, tuple(scores_MLP), bar_width,
   alpha=opacity,
   color='r',
   label='MLP'
)


for rect in rects1:
   ax.text(
      rect.get_x() + rect.get_width()/5,
      1.01*rect.get_height(),
      str(int(rect.get_height())) + "%", 
      color='b', 
      fontweight='bold'
      )

for rect in rects2:
   ax.text(
      rect.get_x() + rect.get_width()/5,
      1.01*rect.get_height(),
      str(int(rect.get_height())) + "%", 
      color='g', 
      fontweight='bold'
      )

for rect in rects3:
   ax.text(
      rect.get_x() + rect.get_width()/5,
      1.01*rect.get_height(),
      str(int(rect.get_height())) + "%", 
      color='y', 
      fontweight='bold'
   )

for rect in rects4:
   ax.text(
      rect.get_x() + rect.get_width()/5,   # label horizontal position
      1.01*rect.get_height(),              # label vertical position
      str(int(rect.get_height())) + "%", 
      color='r', 
      fontweight='bold'
   )

plt.ylim([0, 100])
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title("Scores by model {}/{} dataset".format(training_data_filename, test_data_filename))
plt.xticks(index + bar_width * 1.5, ('Testing', 'Train Test Split', 'Cross Validation'))
plt.legend()
plt.tight_layout()

# plt.plot(k_n_range, knn_scores)
# plt.xlabel("K")
# plt.ylabel("Score")
plt.show()




