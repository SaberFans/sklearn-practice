import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

plt.style.use('ggplot')

titanic = pd.read_csv('titanic dataset.csv')

# pre-processing gender to get numeric
titanic.Sex = titanic.Sex.apply(lambda sex: 1 if sex=='male' else 0)

missing_stats = titanic.isnull().sum()

print('missing stats: \n', (missing_stats))
print('Number of NaN value for passenger age: \n',(missing_stats.get('Age')))

# fill in the missing age
titanic.Age = titanic.Age.fillna(titanic.Age.mean())

print('Number of NaN value for passenger Cabin type: \n', str(missing_stats.get('Cabin')))

# fill in the missing fare
titanic.Fare = titanic.fillna(titanic.Fare.mean())

# try k-nearest neighbors on titanic data

from sklearn.neighbors import KNeighborsClassifier

# instantiate the estimator
knn = KNeighborsClassifier(n_neighbors=5)

# choosing only 4 features,

feature_data = titanic.loc[:, ['Fare', 'Age', 'Sex', 'Pclass']]

target_data = titanic.loc[:, 'Survived']

knn.fit(feature_data, target_data)

# get titanic feature data

scores = cross_val_score(knn, feature_data, target_data, cv = 10)

# print 10 fold cross validation scores
print('10 fold cross validation scores: \n', scores)

# print accuracy of 10 fold cross validation
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# get survived data
survived_data = titanic.loc[titanic['Survived']==1]

print('survived people number is : ',(survived_data.count()[0]))





