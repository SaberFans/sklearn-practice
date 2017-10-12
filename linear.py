
#importing the libraries
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Try put this into a config
pathtocsv = "sum_noise.csv"

#importing the dataset
dataset = pd.read_csv(pathtocsv, delimiter=";")

#Note calls to head and count are not needed are simply for exploration
dataset.head(n=5)

#Get the first 10000 rows
sampledataset10000 = dataset[:10000]
sampledataset10000.count()

#Get some test data
testdataset10000 = dataset[10000:20000]

#separate into features and target
#this is not a copy of the original set
features_set = sampledataset10000.iloc[:,1:11]
features_set.head()
targets_set = sampledataset10000.iloc[:,11]
targets_set.head()
features_test = testdataset10000.iloc[:,1:11]
targets_test = testdataset10000.iloc[:,11]

#Create a linear regression object
linregr = linear_model.LinearRegression()

#Train the model
linregr.fit(features_set, targets_set)

#predict values for the training set
predictedset = linregr.predict(features_test)

#Lets do some cross validation
#I think this is 10 fold
#n_jobs = -1 performs the calculation among all CPUs - not work for me
scores = cross_val_score(linregr, features_set, targets_set, cv = 10)
print('Cross validation scores: \n', scores)

#See what co-efficients were predicted for LR
#There is noise in the data set, but these should be about
#1 1 1 1 0 1 1 1 1 1
print('Coefficients: \n', linregr.coef_)

#How accurate we were based on MSQE
print('Mean squared error: %.2f' % mean_squared_error(targets_test, predictedset))

#Score
print('R^2 score: %.4f' % r2_score(targets_test, predictedset))
#This should be the same as R^2, I am just testing this is the case
print('Default score for linregr %.4f' %linregr.score(features_test, targets_test))

#Plot outputs
plt.scatter(predictedset, targets_test, color = 'black')

plt.xticks(())
plt.yticks(())

plt.show()