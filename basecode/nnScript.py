import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import sys as sys


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    x = 1/(1.0 + np.exp(-z))
    return x



def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    #Pick a reasonable size for validation data


    #Your code here
    validation_label = np.array([])
    train_label = np.array([])
    test_label = np.array([])


    #size of each image matrix
    trainSize0 = mat['train0'].shape[0]
    trainSize1 = mat['train1'].shape[0]
    trainSize2 = mat['train2'].shape[0]
    trainSize3 = mat['train3'].shape[0]
    trainSize4 = mat['train4'].shape[0]
    trainSize5 = mat['train5'].shape[0]
    trainSize6 = mat['train6'].shape[0]
    trainSize7 = mat['train7'].shape[0]
    trainSize8 = mat['train8'].shape[0]
    trainSize9 = mat['train9'].shape[0]

    matType = mat['train0'].dtype

    # training data stacking, type change, and labeling in last column
    train_lab0 =[1,0,0,0,0,0,0,0,0,0]
    train_lab1 =[0,1,0,0,0,0,0,0,0,0]
    train_lab2 =[0,0,1,0,0,0,0,0,0,0]
    train_lab3 =[0,0,0,1,0,0,0,0,0,0]
    train_lab4 =[0,0,0,0,1,0,0,0,0,0]
    train_lab5 =[0,0,0,0,0,1,0,0,0,0]
    train_lab6 =[0,0,0,0,0,0,1,0,0,0]
    train_lab7 =[0,0,0,0,0,0,0,1,0,0]
    train_lab8 =[0,0,0,0,0,0,0,0,1,0]
    train_lab9 =[0,0,0,0,0,0,0,0,0,1]

    a = np.tile(train_lab0, (trainSize0, 1))
    b = np.tile(train_lab1, (trainSize1, 1))
    c = np.tile(train_lab2, (trainSize2, 1))
    d = np.tile(train_lab3, (trainSize3, 1))
    e = np.tile(train_lab4, (trainSize4, 1))
    f = np.tile(train_lab5, (trainSize5, 1))
    g = np.tile(train_lab6, (trainSize6, 1))
    h = np.tile(train_lab7, (trainSize7, 1))
    i = np.tile(train_lab8, (trainSize8, 1))
    j = np.tile(train_lab9, (trainSize9, 1))

    temp_label = np.vstack((a, b))
    temp_label = np.vstack((temp_label, c))
    temp_label = np.vstack((temp_label, d))
    temp_label = np.vstack((temp_label, e))
    temp_label = np.vstack((temp_label, f))
    temp_label = np.vstack((temp_label, g))
    temp_label = np.vstack((temp_label, h))
    temp_label = np.vstack((temp_label, i))
    temp_label = np.vstack((temp_label, j))

    train_data = np.vstack((mat['train0'], mat['train1']))
    train_data = np.vstack((train_data, mat['train2']))
    train_data = np.vstack((train_data, mat['train3']))
    train_data = np.vstack((train_data, mat['train4']))
    train_data = np.vstack((train_data, mat['train5']))
    train_data = np.vstack((train_data, mat['train6']))
    train_data = np.vstack((train_data, mat['train7']))
    train_data = np.vstack((train_data, mat['train8']))
    train_data = np.vstack((train_data, mat['train9']))

    train_data = np.hstack((train_data, temp_label))

 #   print train_data.shape

    # convert the values to type 'double'
    train_data = train_data.astype(np.float64, copy=False)

    # shuffle the matrix
#    map(np.random.shuffle, train_data)
    np.random.shuffle(train_data)

    # split the matrix into training matrix and validation matrix
    train = train_data[0:50000, 0:784]
    validate = train_data[50000:60000, 0:784]

    train_label = train_data[0:50000, 784:795]

    # normalize the training data
    train /= 255
#    print train_label[2000, :]

    # test sizes in test array
    testSize0 = mat['test0'].shape[0]
    testSize1 = mat['test1'].shape[0]
    testSize2 = mat['test2'].shape[0]
    testSize3 = mat['test3'].shape[0]
    testSize4 = mat['test4'].shape[0]
    testSize5 = mat['test5'].shape[0]
    testSize6 = mat['test6'].shape[0]
    testSize7 = mat['test7'].shape[0]
    testSize8 = mat['test8'].shape[0]
    testSize9 = mat['test9'].shape[0]

    matTestType = mat['train0'].dtype

    # training data stacking, type change, and labeling in last column
    test_lab0 =[1,0,0,0,0,0,0,0,0,0]
    test_lab1 =[0,1,0,0,0,0,0,0,0,0]
    test_lab2 =[0,0,1,0,0,0,0,0,0,0]
    test_lab3 =[0,0,0,1,0,0,0,0,0,0]
    test_lab4 =[0,0,0,0,1,0,0,0,0,0]
    test_lab5 =[0,0,0,0,0,1,0,0,0,0]
    test_lab6 =[0,0,0,0,0,0,1,0,0,0]
    test_lab7 =[0,0,0,0,0,0,0,1,0,0]
    test_lab8 =[0,0,0,0,0,0,0,0,1,0]
    test_lab9 =[0,0,0,0,0,0,0,0,0,1]

    a = np.tile(test_lab0, (testSize0, 1))
    b = np.tile(test_lab1, (testSize1, 1))
    c = np.tile(test_lab2, (testSize2, 1))
    d = np.tile(test_lab3, (testSize3, 1))
    e = np.tile(test_lab4, (testSize4, 1))
    f = np.tile(test_lab5, (testSize5, 1))
    g = np.tile(test_lab6, (testSize6, 1))
    h = np.tile(test_lab7, (testSize7, 1))
    i = np.tile(test_lab8, (testSize8, 1))
    j = np.tile(test_lab9, (testSize9, 1))

    test_label = np.vstack((a, b))
    test_label = np.vstack((test_label, c))
    test_label = np.vstack((test_label, d))
    test_label = np.vstack((test_label, e))
    test_label = np.vstack((test_label, f))
    test_label = np.vstack((test_label, g))
    test_label = np.vstack((test_label, h))
    test_label = np.vstack((test_label, i))
    test_label = np.vstack((test_label, j))


    test_data = np.vstack((mat['test0'], mat['test1']))
    test_data = np.vstack((test_data, mat['test2']))
    test_data = np.vstack((test_data, mat['test3']))
    test_data = np.vstack((test_data, mat['test4']))
    test_data = np.vstack((test_data, mat['test5']))
    test_data = np.vstack((test_data, mat['test6']))
    test_data = np.vstack((test_data, mat['test7']))
    test_data = np.vstack((test_data, mat['test8']))
    test_data = np.vstack((test_data, mat['test9']))

    test_data = np.hstack((test_data, test_label))

    # test data stacking, type change, and labeling in last column
    test_data = test_data.astype(np.float64, copy=False)

    #normailize test matrtix
    test_data /= 255

    return train, train_label, validate, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # print "should be (50,785)"
    # print w1.shape
    #
    # print "should be (10,51)"
    # print w2.shape

    #Your code here

    ############################
    # Feed Forward Propagation #
    ############################

    #transpose the training data and labels
    trans_train = training_data.transpose()
    trans_train_labels = training_label.transpose()

    # makes the initial bias set
    bias_train = np.ones((1, 50000))

    #add the bias row to the bottom of the tranposed training data
    trans_train = np.vstack((trans_train,bias_train))

    #calculate the training data
    hidden_layer = np.dot(w1, trans_train)
    hidden_layer = sigmoid(hidden_layer)

    #add bias row to hidden layer
    bias_hidden = np.ones((1, 50000))
#    print bias_hidden
    hidden_layer = np.vstack((hidden_layer, bias_hidden))
#    print hidden_layer.shape
#    print trans_train.shape

    output_layer = np.dot(w2, hidden_layer)
    # for i in range(0,50000):
    #     for j in range(0, 785):
    #         if output_layer[i][j] > 1:
    #             print "out of bounds"
    #             print w2.dtype
    #             print output_layer[i][j]
    #             sys.exit(1)
    output_layer = sigmoid(output_layer)


    ##############################
    # End of Forward Propagation #
    ##############################

    #now need to change labelling to be 1-of-k notation for the error function

    #print trans_train_labels[9, :]
    #sys.exit(1)
    print "HERE"
    J = np.multiply(trans_train_labels, np.log(output_layer))
    K = np.multiply(1-trans_train_labels, np.log(1-output_layer))
    error = J+K
    obj_val = np.sum(error)
    obj_val = -obj_val/train_data.shape[0]

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    ####################
    # Back Propagation #
    ####################

    obj_grad = np.array([])
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)

    # Calculating 8 and 9
    delta2 = output_layer - trans_train_labels
    derW2 = np.dot(delta2, hidden_layer.transpose())

    # Calculating 12 remove the bias
    delta1 = np.multiply((1-hidden_layer), hidden_layer)
    derW1 = np.multiply(delta1, np.dot(w2.transpose(), delta2))
    derW1 = np.dot(derW1, trans_train.transpose())
    derW1 = derW1[0:50, :]

    #Calulating Gradient Error Function
    grad_w1 = grad_w1 + derW1
    grad_w2 = grad_w2 + derW2

    grad_w1 = (grad_w1/train_data.shape[0]) + (lambdaval*w1/train_data.shape[0])
    grad_w2 = (grad_w2/train_data.shape[0]) + (lambdaval*w2/train_data.shape[0])

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    print obj_val
    print obj_grad

    return (obj_val, obj_grad)



def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])

    a = np.dot(data, w1)
    b = sigmoid(a)
    c = np.dot(b, w2)
    d = sigmoid(c)
    labels = np.append(labels, d)

    return labels




"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 1;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
