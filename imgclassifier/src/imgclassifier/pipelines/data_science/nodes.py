from typing import Dict, List
import logging
import numpy as np

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = sigmoid(Z)
        
    cache = (linear_cache, activation_cache)  
        
    return A, cache

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters)//2
    
    # The loop executes the forward propagation across the layer. Each layer outputs A which is passed as an input
    # to the next layer. The input to the first layer is the input X
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "sigmoid")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)
        
    return AL, caches


def compute_cost(AL, Y):
    
    m = Y.shape[1] 
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1./m*np.dot(dZ, A_prev.T)
    db = 1./m*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "sigmoid")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def initialize_param_deep_nn(layer_dims):

    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    
    return parameters

def L_layer_model(layers_dims, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    logger = logging.getLogger(__name__)

    # Parameters initialization.
    parameters_ = initialize_param_deep_nn(layers_dims)

    logger.info("parameters_ %s", str(parameters_))
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        
        AL, caches = L_model_forward(X, parameters_)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters_ = update_parameters(parameters_, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            logger.info("Cost after iteration %i: %f", i, cost)
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    return parameters_

def predict_neural_network_test(flattened_and_Scaled_Data: Dict, parameters_learned: Dict):

    logger = logging.getLogger(__name__)

    X = flattened_and_Scaled_Data["test_x"]
    y = flattened_and_Scaled_Data["test_y"]

    parameters = parameters_learned
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    logger.info("Accuracy against test dataset %f", np.sum((p == y)/m))

    probability_outcomes_test = p

    return probability_outcomes_test

def predict_neural_network_train(flattened_and_Scaled_Data: Dict, parameters_learned: Dict):

    logger = logging.getLogger(__name__)

    X = flattened_and_Scaled_Data["train_x"]
    y = flattened_and_Scaled_Data["train_y"]

    parameters = parameters_learned
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    logger.info("Accuracy against train dataset %f", np.sum((p == y)/m))

    probability_outcomes_train = p

    return probability_outcomes_train

def train_Neural_Network(parameters : Dict, flattened_and_Scaled_Data: Dict) -> Dict:

    layer_dims = list(parameters["nnarchitecture"].split(","))
    layer_dims = [int(i) for i in layer_dims]
    train_x = flattened_and_Scaled_Data["train_x"]
    train_y = flattened_and_Scaled_Data["train_y"]
    iterations = parameters["iterations"]

    parameters_learned = L_layer_model(layer_dims, train_x, train_y, num_iterations = iterations, print_cost = True)

    return parameters_learned
