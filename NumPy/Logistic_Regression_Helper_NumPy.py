#Arpan Bag

#Helper file that contains all the required user defined functions for Logistic Regression



## Initialization
import numpy as np





#=============================== Cost Function ===============================
#COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   J = COSTFUNCTIONREG(theta, X, Y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters. 


def costFunctionReg(theta, X, Y, lambda_):
	
	# Initialize some useful values
	m = len(Y) # number of training examples

	J = 0	#Cost
	grad = np.zeros((theta.shape))
	
	# =========================== CODE HERE =========================
	# Instructions: Compute the cost of a particular choice of theta.
	#               Set J to the cost.
	#               Compute the partial derivatives and set grad to the partial
	#               derivatives of the cost w.r.t. each parameter in theta

	h_ = sigmoid(np.dot(X,theta))
	J = 1/m*(np.dot(-Y.transpose(),np.log(h_))-np.dot((1-Y).transpose(),np.log(1-h_))) + lambda_/(2*m)*np.sum(np.square(theta[1:]))

	first_weight = (1/m*np.dot(X.transpose(),(h_-Y)))
	rest_weights = (1/m*np.dot(X.transpose(),(h_-Y)) + lambda_/m*theta)
	
	grad[0] = first_weight[0]	#No need to use regularization for the first weight
	grad[1:] = rest_weights[1:]
	
	return J,grad





#=============================== Sigmoid Function ===============================
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
















