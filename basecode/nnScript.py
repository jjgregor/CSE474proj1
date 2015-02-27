import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


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

    den = 1.0 + np.exp(-1.0 * z)
    x = 1.0 / den
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

    np.hstack((mat['train0'], np.zeros((trainSize0, 1), dtype = matType)))

    # training data stacking, type change, and labeling in last column
    a = np.vstack((np.hstack((mat['train0'], np.zeros((trainSize0, 1), dtype = matType))),
                   np.hstack((mat['train1'], np.ones((trainSize1, 1), dtype = matType)))))
    b = np.vstack((a, np.hstack((mat['train2'], 2*np.ones((trainSize2, 1), dtype = matType)))))
    c = np.vstack((b, np.hstack((mat['train3'], 3*np.ones((trainSize3, 1), dtype = matType)))))
    d = np.vstack((c, np.hstack((mat['train4'], 4*np.ones((trainSize4, 1), dtype = matType)))))
    e = np.vstack((d, np.hstack((mat['train5'], 5*np.ones((trainSize5, 1), dtype = matType)))))
    f = np.vstack((e, np.hstack((mat['train6'], 6*np.ones((trainSize6, 1), dtype = matType)))))
    g = np.vstack((f, np.hstack((mat['train7'], 7*np.ones((trainSize7, 1), dtype = matType)))))
    h = np.vstack((g, np.hstack((mat['train8'], 8*np.ones((trainSize8, 1), dtype = matType)))))
    i = np.vstack((h, np.hstack((mat['train9'], 9*np.ones((trainSize9, 1), dtype = matType)))))

    train_data = i

    # convert the values to type 'double'
    train_data = train_data.astype(np.float64, copy=False)

    # normalize the training data
    train_data[:,:-1] /= 255

    # shuffle the matrix
#    map(np.random.shuffle, train_data)
    np.random.shuffle(train_data)

    # split the matrix into training matrix and validation matrix
    train = train_data[0:50000, 0:785]
    validate = train_data[50000:60000, 0:785]

    for x in range(0,50000):
        train_label = np.append(train_label, train_data[x, 784])

    for x in range(0, 10000):
        validation_label = np.append(validation_label, train_data[x,784])


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

    np.hstack((mat['test0'], np.zeros((testSize0, 1), dtype = matTestType)))

    # test data stacking, type change, and labeling in last column
    a = np.vstack((np.hstack((mat['test0'], np.zeros((testSize0, 1), dtype = matTestType))),
                   np.hstack((mat['test1'], np.ones((testSize1, 1), dtype = matTestType)))))
    b = np.vstack((a, np.hstack((mat['test2'], 2*np.ones((testSize2, 1), dtype = matTestType)))))
    c = np.vstack((b, np.hstack((mat['test3'], 3*np.ones((testSize3, 1), dtype = matTestType)))))
    d = np.vstack((c, np.hstack((mat['test4'], 4*np.ones((testSize4, 1), dtype = matTestType)))))
    e = np.vstack((d, np.hstack((mat['test5'], 5*np.ones((testSize5, 1), dtype = matTestType)))))
    f = np.vstack((e, np.hstack((mat['test6'], 6*np.ones((testSize6, 1), dtype = matTestType)))))
    g = np.vstack((f, np.hstack((mat['test7'], 7*np.ones((testSize7, 1), dtype = matTestType)))))
    h = np.vstack((g, np.hstack((mat['test8'], 8*np.ones((testSize8, 1), dtype = matTestType)))))
    i = np.vstack((h, np.hstack((mat['test9'], 9*np.ones((testSize9, 1), dtype = matTestType)))))

    test_data = i
    test_data = test_data.astype(np.float64, copy=False)

    #normailize test matrtix
    test_data[:,:-1] /= 255

    for x in range(0, 10000):
        test_label = np.append(test_label, test_data[x,784])

    return train, train_label, validate, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
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

    #should be (50,785)
    print w1.shape

    #should be (10,51)
    print w2.shape

    #Your code here

    #transpose the training data and labels
    trans_train = training_data.transpose()
    trans_train_labels = training_label.transpose()

    #should be (784,50000)
    print trans_train.shape
    #should be (784,1)
    print trans_train_labels.shape

    # makes the initial bias set
    bias_train = np.ones(1,50000)

    #add the bias row to the bottom of the tranposed training data
    trans_train = np.vstack((trans_train,bias_train))

    #shape should be (785,50000)
    print trans_train.shape

    #calculate the training data
    hidden_layer = w1 * trans_train
    hidden_layer = sigmoid(hidden_layer)

    #should be (50,50000)
    print hidden_layer.shape

    #add bias row to hidden layer
    bias_hidden = np.ones(1,50000)
    hidden_layer = np.vstack((hidden_layer,bias_hidden))

    output_layer = w2* hidden_layer






    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])

    return (obj_val,obj_grad)



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
    #Your code here

    for j in range(w2.shape[1]):
        sum = 0.0
        for i in range(w1.shape[1]):
            sum += data[i] * w1[i][j]
        sigmoid(sum)

    for k in range(0, 10000):
        sum = 0.0
        for j in range(w2.shape[1]):
            sum += data[j] * w2[j][k]
        labels = np.append(sigmoid(sum))
    print labels.shape
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
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
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
