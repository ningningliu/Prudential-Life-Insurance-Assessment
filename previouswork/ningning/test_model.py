import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
def loader(path):
    file = pd.read_csv(path)
    return file

train_data= loader('train.csv')
target = train_data['Response']
test_data = loader('test.csv')
#data = train_data.append(test_data)
target_mean= target.mean()
target_var = target.std(ddof=1)
# Preprocess data #

# factorize categorical variables
train_data['Product_Info_2'] = pd.factorize(train_data['Product_Info_2'])[0]
test_data['Product_Info_2'] = pd.factorize(test_data['Product_Info_2'])[0]

# drop id variable
train_data = train_data.drop('Id',axis =1)
test_data = test_data.drop('Id',axis =1)

#feature scaling and standardisation/ normalisation 

def feature_scale_exDummy(df):
    dummy =[]
    for i in range (48):
        i +=1
        dummy.append('Medical_Keyword_'+ str(i))
    for var in df.columns:
        if var in dummy:
            pass
        else:
            df[var]=(df[var]-df[var].mean())/df[var].std(ddof=1)
    return df

data = feature_scale_exDummy(train_data)
test = feature_scale_exDummy(test_data)

# extract X and y
X= data.drop('Response', axis =1)
y= data['Response']

# extract binary variable
keyWords_list = []
for i in range (48):
    i +=1
    keyWords_list.append('Medical_Keyword_'+ str(i))

X['keyword_count']= X[keyWords_list].sum(axis=1)
X= X.drop(X[keyWords_list],axis =1)
X['BMI_Age'] = X['BMI']*X['Ins_Age']

test['keyword_count']= test[keyWords_list].sum(axis=1)
test= test.drop(test[keyWords_list],axis =1)
test['BMI_Age'] = test['BMI']*test['Ins_Age']

# fill in mean in Nan
train= X.fillna(X.mean())
test= test.fillna(X.mean())


# Model testing 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train,y)
pred = regressor.predict(train)

# denomalise
def denorm(y):
    de_y = y*target_var+target_mean
    return pd.DataFrame(de_y)

de_pred = denorm(pred)
test_pred = denorm(regressor.predict(test))
y_target = pd.DataFrame(target.values).astype(int)

# cut prediction into 8 categories
from sklearn.neighbors import KNeighborsClassifier
classifier_K = KNeighborsClassifier(n_neighbors =5 ,metric='minkowski',p=2 )
classifier_K.fit(de_pred,y_target)
prediction_result = classifier_K.predict(test_pred)
Id_dt = loader('test.csv')
Id = pd.DataFrame(Id_dt['Id'])

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators= 10, criterion='entropy',random_state=0)
classifier_rf.fit(de_pred,y_target)
prediction_result = classifier_rf.predict(test_pred)

prediction=Id
prediction['Response']= pd.DataFrame(prediction_result)
prediction.to_csv('predication.csv', index = False)


# compare data with original distribution of response categories

plt.style.use('ggplot')

target_dis = target.value_counts().sort_index()
pred_dis = pd.Series(prediction_result).value_counts().sort_index()

plt.figure(figsize=(20, 10))
df_plot = pd.concat([target_dis, pred_dis], axis=1)
df_plot.columns = ['train_target','test_prediction']
df_plot.plot.bar()
plt.show()



