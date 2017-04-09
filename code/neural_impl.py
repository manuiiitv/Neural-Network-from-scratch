import numpy as np
from numpy import *
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import matplotlib
import time
import tensorflow as tf
np.random.seed(0)
#X, y = make_circles(40, noise=0.001)
#X, y = make_blobs(30,n_features=2, centers=3)
X = np.array([[0, 0],[0, 1], [1, 0 ],[1, 1]])
y = np.array([0, 1, 1, 0])
#print(X)
#print(y)
#print(X.shape)
#print(y.shape)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()
print(X)
print(X.shape)
print(y)
print(y.shape)

#X = np.multiply(X,X)

#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral, hold = 'True')
#plt.show()
num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2# output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.09 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    #a1 = tf.nn.relu(z1)

    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(nn_hdim, num_passes=200001, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
   
   
    b1 = np.zeros((1, nn_hdim))
 
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
   
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
       
        # Forward propagation
        z1 = X.dot(W1) + b1
        
        a1 = np.tanh(z1)
        
        z2 = a1.dot(W2) + b2
        
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 100 == 0:
          print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))
        # Set min and max values and give it some padding
        if i%1 == 0: 
          x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
          y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
          h = 0.01
    # Generate a grid of points with distance h between them
          xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
          Z = predict(model,np.c_[xx.ravel(), yy.ravel()])
          Z = Z.reshape(xx.shape)
       
          plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
          plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral, hold = 'True')
          plt.pause(0.0005)
          #plt.show()
   
    return model

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.

model = build_model(50, print_loss=True)


