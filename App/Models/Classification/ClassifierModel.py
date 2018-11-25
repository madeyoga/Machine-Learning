import sys
import os 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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
        
        self.model = MLPClassifier(
            solver='lbfgs',
            alpha=1e-5, # 5 nol dibelakang koma, 0.00005 # learning rate
            activation='relu', # perceptron activation relu
            hidden_layer_sizes=(5, 2), 
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
