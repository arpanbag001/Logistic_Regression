##Arpan Bag
##Logistic Regression

## Initialization
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb



np.set_printoptions(formatter={'float_kind':'{:f}'.format})	#Display numbers normally, not using exponent.






print("*****Logistic Regression*******\n\n\n")
input("Press enter to select the Input data file.\n")



## Load Data


print('Loading data ...\n')
data = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\" +"Sample_Data_2.txt",delimiter=",") #The file containing the training data. The Last column is the output (Y) and the rest of the columns are input (X) 

X = data[:,:-1] #Inputs
Y = data[:,-1].reshape(-1, 1)	 #Outputs
m = len(Y);		 #Number of training examples
num_features = X.shape[1]	 #No of features, which is the dimension of X
#Print out some data points
print('\nData loaded.\nFirst 10 examples from the dataset: \n')
print(np.column_stack((X,Y))[0:10,:])


if(num_features == 2):		#Check if the input is two dimensional
	#Plot data
	print("\nPress enter to plot data.\n")
	input()
	
	mask = Y == 1
	pos = plt.scatter(np.extract(mask,X[:,0]), np.extract(mask,X[:,1]), marker="+")
	neg = plt.scatter(np.extract(~mask,X[:,0]), np.extract(~mask,X[:,1]), marker=".")
	
	plt.xlabel("Feature 1 (x1)")
	plt.ylabel("Feature 2 (x2)")
	
	plt.legend((pos,neg),("y = 1", "y = 0"))
	
	plt.show(block=False)
	
	
print("\nProgram paused. Press enter to continue.\n")
input()




	











































