# train test split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from sklearn.model_selection import train_test_split
df = pd.read_csv('cleaned_data1.csv',sep=',')


feature_col = df.columns[:-1]
X = df[feature_col]
Y = df['Response']

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# model
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

model = LinearRegression()
model.fit(X, Y)
pred = model.predict(data)

# cutting point model

from sklearn.linear_model import LogisticRegression
pred = pd.DataFrame(pred)
y_target = pd.DataFrame(Y.values).astype(int)
classifier = LogisticRegression ()
classifier.fit(pred, y_target)
prediction_result = classifier.predict(pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_target, prediction_result)

from sklearn.neighbors import KNeighborsClassifier
classifier_K = KNeighborsClassifier(n_neighbors =5 ,metric='minkowski',p=2 )
classifier_K.fit(pred,y_target)
prediction_result = classifier_K.predict(pred)
cm = confusion_matrix(y_target, prediction_result)

from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear', random_state=0)
classifier_svm.fit(pred,y_target)
prediction_result = classifier_svm.predict(pred)

from sklearn.naive_bayes import GaussianNB
classifier_bs = GaussianNB()
classifier_bs.fit(pred,y_target)
prediction_result = classifier_bs.predict(pred)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators= 10, criterion='entropy',random_state=0)
classifier_rf.fit(pred,y_target)
prediction_result = classifier_rf.predict(pred)


# compare data with original distribution of response categories

plt.style.use('ggplot')

target_dis = Y.value_counts().sort_index()
pred_dis = pd.Series(prediction_result).value_counts().sort_index()

plt.figure(figsize=(20, 10))
df_plot = pd.concat([target_dis, pred_dis], axis=1)
df_plot.columns = ['target','pred']
df_plot.plot.bar()
plt.show()

