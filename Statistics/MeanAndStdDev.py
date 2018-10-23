import math
import csv
import pandas as pd

def loadCSV(filename):
    with open (filename, 'r', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)

        dataset.pop(0) # pop header
        for i in range(len(dataset)):
            dataset[i] = [float(number) for number in dataset[i]]
        return dataset

class Statistics:
    def get_mean(self, data_set):
        return sum(data_set)/len(data_set)

    def get_standard_devations(self, data_set):
        mean = self.get_mean(data_set)
        variance = 0
        for data in data_set:
            variance += math.pow(data - mean, 2)
        st_dev = math.sqrt((1/(len(data_set) - 1)) * variance)
        return st_dev

    def get_variance(self, data_set):
        return math.pow(self.get_standard_devations(data_set), 2)
    
    def get_updated_value(self, xi, mean, st_dev):
        return (xi - mean) / st_dev

    def get_correlation_coefficient(self, data_set):
        E = 0
        
        x_values = []
        y_values = []
        for data in data_set:
            x_values.append(data[0])
            y_values.append(data[1])
        
        mean_x = self.get_mean(x_values)
        stdev_x = self.get_standard_devations(x_values)
        mean_y = self.get_mean(y_values)
        stdev_y = self.get_standard_devations(y_values)

        for i in range(len(data_set)):
            E += self.get_updated_value(x_values[i], mean_x, stdev_x) * self.get_updated_value(y_values[1], mean_y, stdev_y)
        r = 1/len(data_set) * E
        # r = 0
        return r

st = Statistics()
traningSet = loadCSV('G:\\Programs\\Python\\Machine Learning\\Statistics\\sample.csv')
traningSet.pop(0)
# traningSet = pd.read_csv('G:\Programs\Python\Machine Learning\Statistics\\sample.csv')
# traningSet_dict = dict(traningSet)
print(traningSet)
print(st.get_correlation_coefficient(traningSet))


# set1 = [1, 4, 3, 5, 1]
# set2 = [50, 52, 48, 48, 50]
# set3 = [90, 90, 90, 90, 90]
# set4 = [15, 40, 10, 18, 31]
# print("set1: "+str(get_standard_devations(set1)))
# print("set2: "+str(get_standard_devations(set2)))
# print("set3: "+str(get_standard_devations(set3)))
# print("set4: "+str(get_standard_devations(set4)))
        