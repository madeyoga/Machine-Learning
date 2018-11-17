import os
import pandas as pd

from Models.Classification.ClassifierModel import NBGaussModel, KNeighborsModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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


        
