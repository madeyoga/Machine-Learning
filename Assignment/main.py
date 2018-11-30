
import os
import pandas as pd
import sys

import numpy as np

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class ClassifierModel:
    def __init__(self):
        self.model = None
        self.current_predicted_y = None
        self.current_accuracy = -1

    def train(self, dataset=None, X=None, Y=None):
        if not dataset is None and not dataset.empty:
            X = dataset.loc[:, dataset.columns[:-1]]
            Y = dataset.loc[:, dataset.columns[-1:]]
        try:
            self.model.fit(X, Y.values.ravel())
        except ValueError as error:
            raise error.with_traceback(sys.exc_info()[2])

    def predict_y(self, test_set=None, test_set_x=None):
        if test_set and not test_set.empty:
            test_set_x = test_set.loc[:, test_set.columns[:-1]]
        self.current_predicted_y = self.model.predict(test_set_x)
        return self.current_predicted_y

    def get_predicted_y(self):
        return self.current_predicted_y
    
    def get_accuracy(self, Y_test=None):
        try:
            self.current_accuracy = accuracy_score(Y_test, self.current_predicted_y)
        except ValueError as error:
            error.with_traceback(sys.exc_info()[2])
        return self.current_accuracy
    
    def get_confusion_mat(self, Y_test=None):
        return confusion_matrix(self.current_predicted_y, Y_test)

    def get_f1_score(self, Y_test=None):
        return f1_score(self.current_predicted_y, Y_test)

class DTreeModel(ClassifierModel):
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=0)

class MultiLayerPerceptronModel(ClassifierModel):
    """ 
    Multi-layer is sensitive to feature scaling, 
    so it is highly recommended to scale data.
    scale each attribute to [0, 1] or [-1, +1]
    or standardize it to have mean 0 and variance 1.
    - activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    - solver     : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
        The solver for weight optimization.
        ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
        ‘sgd’ refers to stochastic gradient descent.
        ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
    """
    def __init__(self):
        
        self.model = MLPClassifier (
            solver='lbfgs',
            alpha=1e-5, # 5 nol dibelakang koma, 0.00005 # learning rate
            activation='relu', # perceptron activation relu
            hidden_layer_sizes=(5, 2), # tuple (n perceptron unit, n hidden layer)
            random_state=1,
        )

class NBGaussModel(ClassifierModel):
    def __init__(self):
        self.model = GaussianNB()

class KNeighborsModel(ClassifierModel):
    def __init__(self, n=5):
        self.model = KNeighborsClassifier(n_neighbors=n)
    
    def set_k(self, n=5):
        self.model = KNeighborsClassifier(n_neighbors=n)


class Manager:
    def __init__(self, training_dataset=None, test_dataset=None):
        
        if training_dataset.empty or test_dataset.empty:
            abspath             = os.path.abspath(__file__)
            this_script_path    = os.path.dirname(abspath)
            datasets_path       = this_script_path + "\\Datasets"
            os.chdir(datasets_path)

            training_dataset    = pd.read_csv("iris-train.csv")
            test_dataset        = pd.read_csv("iris-test.csv" )
        
        self.training_dataset   = training_dataset
        self.X_train            = training_dataset.loc[:, training_dataset.columns[:-1]]
        self.Y_train            = training_dataset.loc[:, training_dataset.columns[-1:]]

        self.test_dataset       = test_dataset
        self.X_test             = test_dataset.loc[:, test_dataset.columns[:-1]]
        self.Y_test             = test_dataset.loc[:, test_dataset.columns[-1:]]
    
    def scale_data(self):
        self.X_train = MinMaxScaler().fit_transform(self.X_train)
        self.X_test  = MinMaxScaler().fit_transform(self.X_test)

        self.X_train = pd.DataFrame(
            data=self.X_train[0:,0:]
        )

        self.X_test = pd.DataFrame(
            data=self.X_test[0:,0:]
        )

    def do_train_test_split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_train,
            self.Y_train,
            test_size=0.3,
            random_state=4
        )
        
    def set_training_dataset(self, training_dataset):
        self.training_dataset   = training_dataset
        self.X_train            = training_dataset.loc[:, training_dataset.columns[:-1]]
        self.Y_train            = training_dataset.loc[:, training_dataset.columns[-1:]]
    
    def set_test_dataset(self, test_dataset):
        self.test_dataset       = test_dataset
        self.X_test             = test_dataset.loc[:, test_dataset.columns[:-1]]
        self.Y_test             = test_dataset.loc[:, test_dataset.columns[-1:]]


"""
* Model : KNeighbors, Gaussian Naive Bayes, 
* Neural Network model sangat sensitive pada data yg valuenya tidak di scale.
* Cross Validation score, .mean()
* Cross Validation, menggunakan Startified KFold, variasi dari KFold
"""

abspath             = os.path.abspath(__file__)
this_script_path    = os.path.dirname(abspath)
datasets_path       = this_script_path + "\\Datasets"

os.chdir(datasets_path)

training_data_filename  = "heartdisease-train.csv"
test_data_filename      = "heartdisease-test.csv"  

# training_data_filename  = input("input training dataset file name: ")
# test_data_filename      = input("input test dataset file name : ")

training_dataset    = pd.read_csv(training_data_filename)
test_dataset        = pd.read_csv(test_data_filename)

print("Load Dataset... ({}/{})".format(training_data_filename, test_data_filename))
manager = Manager(training_dataset=training_dataset, test_dataset=test_dataset)
manager.scale_data()

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

scores_KNN = []
scores_GNB = []
scores_DT  = []
scores_MLP = []

## PREDICT/TRAINING
GNB_model.predict_y(test_set_x=manager.X_train)
KNN_model.predict_y(test_set_x=manager.X_train)
DT_model. predict_y(test_set_x=manager.X_train)
MLP_model.predict_y(test_set_x=manager.X_train)
print("Training Accuracy: ")
print("KNN : {}%".format(KNN_model.get_accuracy(Y_test=manager.Y_train) * 100))
print("GNB : {}%".format(GNB_model.get_accuracy(Y_test=manager.Y_train) * 100))
print("DT  : {}%".format(DT_model .get_accuracy(Y_test=manager.Y_train) * 100))
print("MLP : {}%".format(MLP_model.get_accuracy(Y_test=manager.Y_train) * 100))

scores_KNN.append(KNN_model.current_accuracy * 100)
scores_GNB.append(GNB_model.current_accuracy * 100)
scores_DT .append(DT_model .current_accuracy * 100)
scores_MLP.append(MLP_model.current_accuracy * 100)

## PREDICT/TESTING
GNB_model.predict_y(test_set_x=manager.X_test)
KNN_model.predict_y(test_set_x=manager.X_test)
DT_model. predict_y(test_set_x=manager.X_test)
MLP_model.predict_y(test_set_x=manager.X_test)

print("Test Accuracy: ")
print("CONFUSION MATRIX :")
print(GNB_model.get_confusion_mat(Y_test=manager.Y_test))
print(KNN_model.get_confusion_mat(Y_test=manager.Y_test))
print(DT_model .get_confusion_mat(Y_test=manager.Y_test))
print(MLP_model.get_confusion_mat(Y_test=manager.Y_test))

print("F1-Score :")
print(GNB_model.get_f1_score(Y_test=manager.Y_test) * 100)
print(KNN_model.get_f1_score(Y_test=manager.Y_test) * 100)
print(DT_model .get_f1_score(Y_test=manager.Y_test) * 100)
print(MLP_model.get_f1_score(Y_test=manager.Y_test) * 100)

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
scores_GNB.append(GNB_model.get_accuracy(Y_test=manager.Y_test) * 100)
scores_DT .append(DT_model .get_accuracy(Y_test=manager.Y_test) * 100)
scores_MLP.append(MLP_model.get_accuracy(Y_test=manager.Y_test) * 100)

print("Training Accuracy with train_test_split approach: ")
print("KNN : {}%".format(max(knn_scores) * 100))
print("GNB : {}%".format(GNB_model.current_accuracy * 100))
print("DT  : {}%".format(DT_model .current_accuracy * 100))
print("MLP : {}%".format(MLP_model.current_accuracy * 100))

#################################
### CROSS VALIDATION APPROACH ###
#################################
manager = Manager(training_dataset=training_dataset, test_dataset=test_dataset)
manager.scale_data()

print("Training Accuracy with cross validation approach: ")
cvs_knn = cross_val_score(KNN_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100
cvs_gnb = cross_val_score(GNB_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100
cvs_dt  = cross_val_score(DT_model .model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100
cvs_mlp = cross_val_score(MLP_model.model, manager.X_train, manager.Y_train.values.ravel(), cv=10).mean() * 100
print(cvs_knn)
print(cvs_gnb)
print(cvs_dt)
print(cvs_mlp)
scores_KNN.append(cvs_knn)
scores_GNB.append(cvs_gnb)
scores_DT .append(cvs_dt)
scores_MLP.append(cvs_mlp)


from matplotlib import pyplot as plt

n_groups = 4

ax = plt.subplot()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.5

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
plt.xlabel('Context')
plt.ylabel('Scores')
plt.title("Scores by model {}/{} dataset".format(training_data_filename, test_data_filename))
plt.xticks(index + bar_width * 1.5, ('Training', 'Testing', 'Train Test Split', 'Cross Validation'))
plt.legend()
plt.tight_layout()

# plt.plot(k_n_range, knn_scores)
# plt.xlabel("K")
# plt.ylabel("Score")
plt.show()
