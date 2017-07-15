import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Load data file

class DataLoader:
    def __init__(self, path):
        self.path = path
        # print('in DataLoader')

    def loader(self):
        # print('Loading data.')
        file = pd.read_csv(self.path)
        # print('Finish loading')
        return file

# load a part of the data
train = DataLoader(path='train_small.csv')
train_data= train.loader()

# SVM
cols = train_data.columns  # features
# transfer dataframe to matrix
train_data = train_data[list(cols)].values
y = train_data[:,len(cols)-1]
X = train_data[:, 0:(len(cols)-1)]
# X1 = X[0:5,:]
#

# print(y)
# print(X)
# print(X1)
#
#
#
"""
clf = SVC(C=0.5, kernel='rbf')
# cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X, y)
print(clf.score(X,y)) # Returns the mean accuracy on the given test data and labels.
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))

y1 = clf.predict(X1)
print(X1)
print(y1)
"""
# find a better C

C = 0.1
best_C = 0.1
best_score = 0
while (C<=2):
    clf = SVC(C=C, kernel='rbf')
    clf.fit(X,y)
    score = clf.score(X,y)
    if score > best_score:
        best_C = C
        best_score = score
    C += 0.1

print(best_C)
print(best_score)

