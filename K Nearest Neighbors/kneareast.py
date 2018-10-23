import csv
import os

import math
import random
import operator

os.chdir(r'g:\\Programs\\python\\Machine Learning\\K Nearest Neighbors')

testSet = []
trainingSet = []

def loadTrainingSet(filename):
    with open (filename, 'r', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range (len(dataset)):
            for j in range (len(dataset[i])):
                dataset[i][j] = float(dataset[i][j])
            trainingSet.append(dataset[i])
            
def loadTestSet(filename):
    with open (filename, 'r', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        # converts traningset data to float
        for i in range (len(dataset) - 1):
            for j in range (len(dataset[i])):
                dataset[i][j] = float(dataset[i][j])
            testSet.append(dataset[i])

def euclideanDistance(data1, data2, data_length):
    distance = 0
    for i in range (data_length):
        distance += pow(float(data1[i]) - float(data2[i]), 2)
    return math.sqrt(distance)

def getNeighbors(testInstance, k):
    # GET ALL TRANINGSET DATA 'DISTANCES' TO TESTINSTANCE
    distances = []
    length = len(testInstance) - 1
    for training_data in trainingSet:
        distance = euclideanDistance(testInstance, training_data, length)
        distances.append((training_data, distance))
    distances.sort(key=operator.itemgetter(1))

    # GET K SMALLEST DISTANCE FROM 'NEIGHBORS'
    neighbors = []
    for i in range (k):
        neighbors.append(distances[i])
    return neighbors

def getResponse(neighbors):
    instanceVotes = {}

    for neighbor in neighbors:
        # [1] is the DISTANCE VALUE
        response = neighbor[0][-len(neighbors)] 
        if response in instanceVotes:
            instanceVotes[response] += 1
        else:
            instanceVotes[response] = 1
    
    sortedVotes = sorted(instanceVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range (len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct+=1
            print(correct)
    return (correct/float(len(testSet))) * 100

def main():
    loadTrainingSet("heartdisease-train.csv")
    loadTestSet("heartdisease-test.csv")
    predictions = []
    for x in range(len(testSet)):
	    neighbors = getNeighbors(testSet[x], 1)
	    result = getResponse(neighbors)
	    predictions.append(result)
	    print('prediksi: ' + repr(int(result)) + ', actual: ' + repr(testSet[x][-1]))
    acc = getAccuracy(testSet, predictions)
    print(str(acc) + "%")

main()

    