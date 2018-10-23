import math

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
