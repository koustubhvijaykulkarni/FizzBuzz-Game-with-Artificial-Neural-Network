import pandas as pd
import pandas_ml as pd1

def fizzbuzz(n):
    
    # Logic Explanation- 
    """In this logic based appraoch we are simply checking if no. is divisble by bith 3 & 5 then we are showing output as 'FizzBuzz'.
    If no. is only divisible by only 3 then 'fizz' and if divisble by only 5 then 'buzz' otherwise 'other'.
    """
    if n % 3 == 0 and n % 5 == 0:
        return 'fizzbuzz'
    elif n % 3 == 0:
        return 'fizz'
    elif n % 5 == 0:
        return 'buzz'
    else:
        return 'other'
		
def createInputCSV(start,end,filename):
    
    # Why list in Python?- inputData list will hold values of all the numbers which will be our input
    #-outputData will store output that we got from software 1.0. i.e. 'fizz', 'buzz', 'fizzbuzz' or 'other'
    #We will use these two lists to create our training data and testing data Csv files.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?- we need to train our model and for that we need lots of data so that our model will learn from that labelled data. 
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i)) #Here we are using our logic based(software 1.0 code for creating our training and testing data)
    
    # Why Dataframe?- Dataframe is used to create two-dimensional data structure to store all inputs and labels that we have so that we can create our training and testing CSV files.
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")

def processData(dataset):
    
    # Why do we have to process?
    """-Here we are separating all the inputs and their corresponding labels. 
    We are converting input numbers into equivalent 10 bit binary numbers 
    and converting labels into binary class matrix by using to_categorical method.
    """
    data   = dataset['input'].values
    labels = dataset['label'].values
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel

def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        # Why do we have number 10?- Here we are converting each number into its binary equivalaent.And as numbers till 1000 can be represented by 10 bits in binary hence we have used 10 here.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)
	
from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "fizzbuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])
    return np_utils.to_categorical(np.array(processedLabel),4)
    #Converts a class vector (integers) to binary class matrix
    
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from pandas_ml import ConfusionMatrix

import numpy as np

input_size = 10
drop_out = 0.1
first_dense_layer_nodes  = 300
second_dense_layer_nodes = 4

def get_model():
    
    # Why do we need a model?-We need a model which will learn from set of labelled data and predict the outcome of the previously unseen data.
    # Why use Dense layer and then activation?
    """ Dense layer has a linear operation in which every input is connected to every output by a weight (so there are n_inputs * n_outputs weights)+bias. 
    It is generally followed by a non-linear activation function. We define activation function for that particular Dense layer. As we are using 'Relu' in first dense layer,we are specifying it right after specifying layer. 
    We use activation function to provide non-linearity. Also, some activation function keep computations of neural network bounded and stable.
    """
    # Why use sequential model with layers?
    """Keras has two models- sequntial and functional.The Sequential model is a linear stack of layers.
    Keras functional model is ideal for defining complex models, such as multi-output models, directed acyclic graphs, or models with shared layers.
    As our problem was fairly simple we are using sequntial model with input layer, one hidden layer and output layer.
    """
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    """In this model, we are using 'Reactified Linear Unit'(Relu) as an activation function. Relu simply activates nodes which has positive value.
    Relu is famous because it does not cause vanishing gradient problem.
    """
    
    # Why dropout?-To prevent overfitting , depending on the value we set as dropout -some nodes are randomly not used while training the model.
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?-softmax converts hidden layer ouputs to any value between 0 and 1- we can consider them as probabilities of a bucket
    
    model.summary()#Print the summary representation of your model
    
    # Why use categorical_crossentropy?- crossentropy is used to measure the performance of a classification model whose output is a probability between o to 1.
    #Categorical crossentropy is used for the multi-class classification problems where each example belongs to a single class and as our problem is multiclass problem hence we are using categorical crossentropy.
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "other"
    elif encodedLabel == 1:
        return "fizz"
    elif encodedLabel == 2:
        return "buzz"
    elif encodedLabel == 3:
        return "fizzbuzz"

# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

model = get_model()

validation_data_split = 0.20 #Spliting of training set in training and validation data.
num_epochs = 1000 #Maximum no of epochs
model_batch_size = 128 #Batch size indicates no of samples to be included in single batch
tb_batch_size = 32
early_patience = 100 #Stops when the loss remains contant or stops showing any improvement i.e. for e.g. if this value is set to 100 then if after 100 epochs loss remains same then it will stop the training. 

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset 
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))

wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))


confusion_matrix = ConfusionMatrix(testData['label'],predictedTestLabel)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot()
# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "Kkulkarn")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50288207")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')