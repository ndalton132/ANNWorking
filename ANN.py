import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math


#Returns 1D array, w/ tuple(normalizedArray, character)
def readFromFile(path):
    with open(path, 'r') as file:
        # Initialize an empty list to store characters
        characters_array = []

        # Read each line in the file
        for line in file:
            # Split each line into individual characters and store them in the characters_array
            characters = [char for char in line.strip()]
            characters_array.extend(characters)
        
        array_2d = np.array([characters_array[i:i+63] for i in range(0, len(characters_array), 63)])
        
        normalized_array = [[1 if char == '#' else 0 for char in row] for row in array_2d]
        
        # Return the 2D array
        return normalized_array
    
         
def createData(numHidden, file):
        lettersArray = np.array(readFromFile(file))
        
                
        W1 = np.array([[rand.uniform(0, .1) for i in range(numHidden)] for j in range(63)])
        B1 = np.array([rand.uniform(0, .1) for i in range(numHidden)])
        
        W2 = np.array([[rand.uniform(0,.1) for i in range(7)] for j in range(numHidden)])
        B2 = np.array([rand.uniform(0,.1) for i in range(7)])
        
        return lettersArray, W1, W2, B1, B2
    
    
class ANNUtils:
    
    def __init__(self, WeightMatrix, BiasMatrix, NodeVals, ZVals):
        self.WeightMatrix = WeightMatrix
        self.BiasMatrix = BiasMatrix
        self.NodeVals = NodeVals
        self.ZVals = ZVals
    
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    
    def derivSigmoid(Z):
        return ANNUtils.sigmoid(Z) * (1 - ANNUtils.sigmoid(Z))
    
    def derivCostFunc(predicted, Actual):
        return 2*(predicted - Actual)
    
    def calculateWeight(fromNode, toNode, Cost):
        #print("A: ",A, "DerivSigmoid: ",Z ,"DerivCost: ", Cost)
        #print("Z", Z)
        return np.dot(toNode.NodeVals,ANNUtils.derivSigmoid(fromNode.ZVals)) * Cost
    
    def calculateBias(Cost, Z):
        return ANNUtils.derivSigmoid(Z)*Cost
    
    #Length of row = # of neurons in the input layer
    #Length of col = # of neurons in hidden layer
    #Sum of each input node -> 1 hiddenLayer node
    #  activation(entireInputLayer, tonNode = nextLayerNode index, entier weight matrix, bias)
    def guessVal(outputLayer, Vals):
        guess = max(outputLayer.NodeVals)
        indexGuess = np.where(outputLayer.NodeVals == guess)
        expected = np.array ([])
        for i in range(len(outputLayer)):
            if i == indexGuess:
                np.append(expected, 1)
            else:
                np.append(expected,0)
    
    def forwardPropogateLayer(self, nextLayer):

        ZVals = np.dot(self.NodeVals.reshape(1, -1), nextLayer.WeightMatrix) 

        ZVals += nextLayer.BiasMatrix
        
        ZVals = ZVals.squeeze()
        nextLayer.ZVals = np.array(ZVals)
        
        #Pass all values through sigmoid and update the next layer
        sigmoid_values = ANNUtils.sigmoid(ZVals)
        nextLayer.NodeVals = np.append(nextLayer.NodeVals, sigmoid_values)
        
    
    def backProp(inputLayer, hiddenLayer, outputLayer, expected, learningRate):
        
        outputToHiddenError = 2 * ((outputLayer.NodeVals) - expected)
        hiddenToInputError = 2 * ((np.dot(outputToHiddenError, outputLayer.WeightMatrix.T)))
        hiddenToInputError = hiddenToInputError.squeeze()
        
        #print("Output ", 2 * outputLayer.NodeVals - expected)
        
        outputLayer.WeightMatrix -= (hiddenLayer.NodeVals.reshape(-1, 1) * ANNUtils.derivSigmoid(outputLayer.ZVals).reshape(1,-1) * outputToHiddenError) * learningRate
        hiddenLayer.WeightMatrix -= (inputLayer.NodeVals.reshape(-1, 1) * ANNUtils.derivSigmoid(hiddenLayer.ZVals).reshape(1,-1) * hiddenToInputError) * learningRate
        

        
        outputLayer.BiasMatrix -= (ANNUtils.derivSigmoid(outputLayer.ZVals) * outputToHiddenError) * learningRate
        hiddenLayer.BiasMatrix -= (ANNUtils.derivSigmoid(hiddenLayer.ZVals) * hiddenToInputError) * learningRate

            
    
def train(inputData, inputLayer, hiddenLayer, outputLayer, goalArr, learningRate, debug):
    for i in range(len(inputData)):
        inputLayer.NodeVals = inputData[i]
        
        if debug == 1:
            print(inputLayer.NodeVals,"/n")
        
        inputLayer.forwardPropogateLayer(hiddenLayer)
        
        if debug == 1:
            #format(hiddenLayer.ZVals)
            format(hiddenLayer.NodeVals)
            
        hiddenLayer.forwardPropogateLayer(outputLayer)
        
        if debug == 1:
            #format(outputLayer.ZVals)
            format(outputLayer.NodeVals)
            
        
        idealArr = np.tile(np.eye(7), (3, 1))[:21]

        ANNUtils.backProp(inputLayer, hiddenLayer,outputLayer, idealArr[i], learningRate)
            
        inputLayer.NodeVals = np.array([])
        hiddenLayer.NodeVals = np.array([])
        outputLayer.NodeVals = np.array([])
        inputLayer.ZVals = np.array([])
        hiddenLayer.ZVals = np.array([])
        outputLayer.ZVals = np.array([])

def test(file, inputLayer, hiddenLayer, outputLayer, correctValues, debug):
    inputData = np.array(readFromFile(file))
    
    idealArr = np.tile(np.eye(7), (3, 1))[:21]
    correct = 0
    incorrect = 0
    
    for i in range(len(inputData)):
        inputLayer.NodeVals = inputData[i]
        
        if debug == 1:
            print(inputLayer.NodeVals,"/n")
        
        inputLayer.forwardPropogateLayer(hiddenLayer)
        
        if debug == 1:
            #format(hiddenLayer.ZVals)
            format(hiddenLayer.NodeVals)
            
        hiddenLayer.forwardPropogateLayer(outputLayer)
        
        ideal_index = np.argmax(idealArr[i])

        output_index = np.argmax(outputLayer.NodeVals)
        
        if ideal_index == output_index:
            correct += 1
            if debug == "resultsDetailed":
                print(correctValues[output_index], "is correct!")
        else:
            incorrect += 1
            if debug == "resultsDetailed":
                print(correctValues[output_index], "is incorrect!")
    
        
        
        if debug == 1:
            #format(outputLayer.ZVals)
            format(outputLayer.NodeVals)
            
        inputLayer.NodeVals = np.array([])
        hiddenLayer.NodeVals = np.array([])
        outputLayer.NodeVals = np.array([])
        inputLayer.ZVals = np.array([])
        hiddenLayer.ZVals = np.array([])
        outputLayer.ZVals = np.array([])
        
    accPerc = correct/(correct + incorrect) * 100
    print("Accuracy = ", accPerc,"%") 
    return accPerc
    
             
def format(Array):
    for i in range(len(Array)):
        print("I =" , i, Array[i])
    print("")

        
def checkArray(Array):
    for i in range(len(Array)):
        if i % 7 != 0:
            print(Array[i], end = "")
        else:
            print(Array[i])
     

def main():
    #Number of nodes in the hidden
    #Returns the initial weight matrix and Bias matrix
    trainingFile = "C:\\Users\\nickd\\Downloads\\HW3_Training.txt"
    testingFile = "C:\\Users\\nickd\\Downloads\\HW3_Testing.txt"
    
    epochs = 11
    hiddenLayerNodes = 100
    learningRate = .3
    
    #Debug options
    # 0: Just accuracy percentage
    # 1:  to print before after training/testing
    # resultsDetailed: prints results for each value then the accuracy
    debug = "resultsDetailed"
            
    inputData, W1, W2, B1, B2 = createData(hiddenLayerNodes, trainingFile)

    inputLayer = ANNUtils(WeightMatrix=np.array([]), BiasMatrix=np.array([]), NodeVals=np.array([]), ZVals=np.array([]))

    hiddenLayer = ANNUtils(WeightMatrix=W1, BiasMatrix=B1, NodeVals=np.array([]), ZVals=np.array([]))

    outputLayer = ANNUtils(WeightMatrix=W2, BiasMatrix=B2, NodeVals=np.array([]), ZVals=np.array([]))
    
    correctValues = np.array(["A","B","C","D","E","J","K","A","B","C","D","E","J","K","A","B","C","D","E","J","K"])

    
    for i in range(0,epochs):
        train(inputData, inputLayer, hiddenLayer, outputLayer, correctValues,learningRate, 0)
        
    #(file, inputLayer, hiddenLayer, outputLayer, correctValues, debug)
    accuracy = test(testingFile, inputLayer, hiddenLayer, outputLayer, correctValues, debug)
    
main()
