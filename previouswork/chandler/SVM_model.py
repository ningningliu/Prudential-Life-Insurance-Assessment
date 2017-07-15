import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def expected_matrices(target):
    w, h = 8, 8
    Matrix = [[0 for x in range(w)] for y in range(h)]
    counted = np.bincount(target)[1:]
    for i in range(8):
        Matrix[i][i] = counted[i]
    return Matrix

def estimate_matrices(predict, y_test):
    w, h = 8, 8
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(8):
        index_tmp = np.where(y_test == i + 1)

        # index_tmp = y_test[y_test == i + 1].index
        for j in range(8):
            impute = sum(predict[index_tmp] == j+1)
            Matrix[i][j] = impute
    return Matrix

def weighted_matrices(expected, estimate):
    n = 8
    w, h = 8, 8
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(8):
        for j in range(8):
            Matrix[i][j] = ((estimate[i][j] - expected[i][j])**2)/((n-1)**2)

    return Matrix

def kappa(y_test, y_predict):
    expected = expected_matrices(y_test)
    estimate = estimate_matrices(y_predict, y_test)
    weighted = weighted_matrices(expected, estimate)

    numerator = sum(sum(np.array(estimate) * np.array(weighted)))
    denominator = sum(sum(np.array(expected) * np.array(weighted)))
    kappa = 1 - numerator/denominator

    return kappa
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
# train = DataLoader(path='train_small.csv')
# load the whole training data
train = DataLoader(path='complete_train.csv')
train_data= train.loader()
# print(train_data)
# SVM
cols = train_data.columns  # features
# transfer dataframe to matrix
train_data = train_data[list(cols)].values
y = train_data[1:,len(cols)-1]
X = train_data[1:, 1:(len(cols)-1)]
# X1 = X[0:5,:]
#

# print(y)
# print(X)
# print(X1)
#
X_train = X
y_train = y

test = DataLoader(path='complete_test.csv')
test_data = test.loader()
cols = test_data.columns
test_data = test_data[list(cols)].values
X_test = test_data[1:,1:len(cols)]

#

clf = SVC(C=1, kernel='rbf')
# cross validation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf.fit(X, y)
# print(clf.score(X,y)) # Returns the mean accuracy on the given test data and labels.
clf.fit(X_train,y_train)
# print(clf.score(X_test, y_test))

y_predict = clf.predict(X_test)
y_predict = pd.DataFrame(y_predict)
y_predict.to_csv('submission_file.csv')

# convert to int64
# y_test = y_test.astype(np.int64)
# y_predict = y_predict.astype(np.int64)

# print(y_test, y_predict)
# kappa_value = kappa(y_test, y_predict)
# print(kappa_value)
# print(X1)
# print(y1)

# find a better C
"""""""""
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
"""""""""
