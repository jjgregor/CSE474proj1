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

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  #your code here



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
 #   print train_data.shape

    # convert the values to type 'double'
    train_data = train_data.astype(np.float64, copy=False)
  #  print train_data.dtype
    # normalize the training data
    print train_data[8000,:]
    train_data[:,:-1] /= 255
    print train_data[8000,:]
    
     np.linalg.norm(train_data[:784], axis=0)

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
    print test_data.shape

    test_data = test_data.astype(np.float64, copy=False)
    print test_data.dtype

    np.linalg.norm(test_data[:784], axis=0)


    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])

    #Test Data stacking and type changing
    a = np.vstack((mat['test0'], mat['test1']))
    b = np.vstack((a, mat['test2']))
    c = np.vstack((b, mat['test3']))
    d = np.vstack((c, mat['test4']))
    e = np.vstack((d, mat['test5']))
    f = np.vstack((e, mat['test6']))
    g = np.vstack((f, mat['test7']))
    h = np.vstack((g, mat['test8']))
    i = np.vstack((h, mat['test9']))

    test_data = i

    test_label = np.array([])

    return train_data, train_label, validation_data, validation_label, test_data, test_label




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

    #Your code here
    #
    #
    #
    #
    #



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
