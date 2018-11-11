import os
import pandas as pd

from Models.Classification.ClassifierModel import NBGaussModel, KNeighborsModel
from sklearn.preprocessing import MinMaxScaler

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

        self.KNN_model = KNeighborsModel(n=5)
        self.GNB_model = NBGaussModel()

        self.KNN_model.train(X=self.X_train, Y=self.Y_train)
        self.GNB_model.train(X=self.X_train, Y=self.Y_train)

        self.compareDataFrame = None
    
    def scale_data(self):
        self.X_train = MinMaxScaler().fit_transform(self.X_train)
        self.X_test  = MinMaxScaler().fit_transform(self.X_test)

        self.X_train = pd.DataFrame(
            data=self.X_train[0:,0:]
        )

        self.X_test = pd.DataFrame(
            data=self.X_test[0:,0:]
        )

        self.update_models()
    
    def update_models(self):
        self.KNN_model.train(X=self.X_train, Y=self.Y_train)
        self.GNB_model.train(X=self.X_train, Y=self.Y_train) 

    def set_training_dataset(self, training_dataset):
        self.training_dataset   = training_dataset
        self.X_train            = training_dataset.loc[:, training_dataset.columns[:-1]]
        self.Y_train            = training_dataset.loc[:, training_dataset.columns[-1:]]
    
    def set_test_dataset(self, test_dataset):
        self.test_dataset       = test_dataset
        self.X_test             = test_dataset.loc[:, test_dataset.columns[:-1]]
        self.Y_test             = test_dataset.loc[:, test_dataset.columns[-1:]]

    def get_compare_frame(self):
        knn_pred    = self.get_KNN_prediction()
        gnb_pred    = self.get_GNB_prediction()
        knn_frame   = pd.DataFrame(
            data={'KNN Predictions' : knn_pred}
        )
        gnb_frame   = pd.DataFrame(
            data={'GNB Predictions' : gnb_pred}
        )
        self.compareDataFrame = self.Y_test.join(knn_frame)
        self.compareDataFrame = self.compareDataFrame.join(gnb_frame)
        return self.compareDataFrame

    def set_KNN_K(self, k=5):
        self.KNN_model.set_k(k)
        self.KNN_model.train(X=self.X_train, Y=self.Y_train)

    def get_KNN_prediction(self):
        self.KNN_model.current_predicted_y = self.KNN_model.predict_y(test_set_x=self.X_test)
        return self.KNN_model.current_predicted_y
    
    def get_KNN_accuracy(self):
        return self.KNN_model.get_accuracy(Y_test=self.Y_test)

    def get_GNB_prediction(self):
        self.GNB_model.current_predicted_y = self.GNB_model.predict_y(test_set_x=self.X_test)
        return self.GNB_model.current_predicted_y

    def get_GNB_accuracy(self):
        return self.GNB_model.get_accuracy(Y_test=self.Y_test)
    

