
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data file #
def loader(path):
    file = pd.read_csv(path)
    return file


train_data= loader(path='train.csv')
test_data = loader(path='test.csv')
data = train_data.append(test_data)

# Preprocess data #

# factorize categorical variables
data['Product_Info_2'] = pd.factorize(data['Product_Info_2'])[0]

# drop id variable
data = data.drop('Id',axis =1)

#feature scaling and standardisation/ normalisation 

def feature_scale(df):
    scale_df = (df - df.mean())/df.std(ddof =1)
    return scale_df



def feature_scale_exDummy(df):
    dummy =[]
    for i in range (48):
        i +=1
        dummy.append('Medical_Keyword_'+ str(i))
    for var in data.columns:
        if var in dummy:
            pass
        else:
            df[var]=(df[var]-df[var].mean())/df[var].std(ddof=1)
    return df

# Create list of variable types

cont_variable_list = ['Product_Info_4', 'Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4',
                      'Employment_Info_6','Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                      'Family_Hist_5']


dis_variable_list = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24',
                     'Medical_History_32']

for i in range (48):
    i +=1
    dis_variable_list.append('Medical_Keyword_'+ str(i))


cat_variable_list =[]
for header in data.columns:
    if header in cont_variable_list and dis_variable_list:
        pass
    else:
        cat_variable_list.append(header)

missing_list = ['Employment_Info_1','Employment_Info_4','Employment_Info_6','Family_Hist_2','Family_Hist_3',
                'Family_Hist_4','Family_Hist_5','Insurance_History_5','Medical_History_1','Medical_History_10',
                'Medical_History_15','Medical_History_24','Medical_History_32']


# recommend method : pca, interpolation,svd, boosting     
def drop_response(df):
    df= df.drop('Response', axis =1)
    return df

def fill_avg(df):
    for var in missing_list:
        df[var] = df[var].fillna(df[var].mean())
    return df
    
def drop_col(df):
    df = df.drop(['Medical_History_10','Medical_History_24','Medical_History_32'])
    return df
        

def fill_svd (df):
    col_mean = np.nanmean(df, axis=0, keepdims=1)
    valid = np.isfinite(df)
    df0 = np.where(valid, df, col_mean)
    halt = True
    maxiter =100
    ii = 1
    normlist = []
    while halt == True:
        U, s, V = np.linalg.svd(df0, full_matrices = False)
        s1 = [(i*0 if i <= 30 else i ) for i in s]
        df1 = U.dot(np.diag(s1).dot(V))
        df2= np.where(~valid, df1, df0)
        norm = np.linalg.norm(df2 - df1)
        normlist.append(norm)
#        print(norm)
        df0=df2
        if norm < 0.00001 or ii >= maxiter:
            halt = False
            error = np.nansum((df1-df)**2)
        ii += 1
        print(ii)
    return df2, normlist, error



#norm_data_1= feature_scale(raw_data)
#pre = MissingMethod(norm_data).drop_response()
#pre_svd = MissingMethod(norm_data_1).drop_response()
#pre_svd = MissingMethod(norm_data_1).fill_avg()
#filled_data, norm, error = fill_svd(pre)

"""
# after iterates 1000, error reduce to 75.79
y= []
for i in range(1000):
    y.append(i)
    i +=1
    
plt.scatter(y,norm,color = 'red')


#####################

"""

# Linear Model 1 #
# X: normalise , svd fill in missing data
# y: normalise
# get dataset ready
norm_data = feature_scale(data)
norm_data_dropResponse = drop_response(norm_data)
X_variables = norm_data_dropResponse.columns
data_1, normlist, error = fill_svd(norm_data_dropResponse) # mse << 0.0001
data_1_df = pd.DataFrame(data = data_1[0:59381,0:],columns= X_variables)
y= pd.DataFrame(data=norm_data['Response'][0:59381], columns = ['Response'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_1_df, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train,y_train)
y_pred = regressor1.predict(X_test)
mse1= np.sum((y_test-y_pred)**2)/len(y_pred)# mse =0.681

import statsmodels.formula.api as sm
regressor_OLS_1 = sm.OLS(endog = y, exog = data_1_df).fit()
regressor_OLS_1.summary() # R-sq=0.375

def linearModel(df):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.20, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    mse= np.sum((y_test-y_pred)**2)/len(y_pred)
    return mse

def lin_regressor_summ(X,y):
    X['constant'] = np.ones((len(X),1))
    regressor_OLS = sm.OLS(endog = y, exog = X).fit()
    return regressor_OLS
    
    
    

# Linear Model 2 #
# X: normalise, svd fill missing data , add BMI*age
# y: normalise
data_2 = data_1_df
data_2['BMI_Age']= data_2['BMI']*data_2['Ins_Age']
mse2= linearModel(data_2) # mse = 0.616
regressor_OLS_2 = lin_regressor_summ(data_2, y)
regressor_OLS_2.summary() # R-sq=0.378

# Linear Model 3 #
# X: normalise exclude dummy, svd fill missing data, add BMI*age
# y: normalise
norm_data_exDum = feature_scale_exDummy(data)
norm_data__exDum_dropResponse = drop_response(norm_data_exDum)
data_3, normlist3, error3 = fill_svd(norm_data__exDum_dropResponse)# mse << 0.000
data_3_df = pd.DataFrame(data = data_3[0:59381,0:],columns= X_variables) 
data_3 = data_3_df
data_3['BMI_Age']= data_3['BMI']*data_3['Ins_Age']
mse3= linearModel(data_3) # mse =0.617
regressor_OLS_3 = lin_regressor_summ(data_3, y)
regressor_OLS_3.summary() # R-sq=0.378

# Linear Model 4#
# X: normalise exclude dummy, svd fill missing data, BMI*Age, medical_keyword_count
# y: normalise
keyWords_list = []
for i in range (48):
    i +=1
    keyWords_list.append('Medical_Keyword_'+ str(i))
    
data_4 = data_3_df
data_4['BMI_Age']= data_4['BMI']*data_4['Ins_Age']
data_4['Keyword_count']= data_4[keyWords_list].sum(axis=1)
data_4= data_4.drop('constant',axis =1)
mse4 = linearModel(data_4) # mse=0.617
regressor_OLS_4 = lin_regressor_summ(data_4, y)
regressor_OLS_4.summary() # R-sq =0.377

# Linear Model 5#
# X: normalise exclude dummy, svd filling missing data
# BMI*Age, medical_keyword_count, drop keywords col
data_5= data_4
data_5= data_5.drop(data_5[keyWords_list],axis=1)
mse5 = linearModel(data_5) # mse =0.635
regressor_OLS_5 = lin_regressor_summ(data_5, y)
regressor_OLS_5.summary() # R-sq =0.360

# Linear Model 6#
# X: normalise exclude dummy, svd filling missing data
# BMI*Age, medical_keyword_count, drop keywords col
# drop x col p-value >0.5: Employment_Info_2,Employment_Info_3,Employment_Info_5,  
#Family_Hist_1,InsuredInfo_4 ,Medical_History_34
drop_list_6= ['Employment_Info_2','Employment_Info_3','Employment_Info_5',  
              'Family_Hist_1','InsuredInfo_4' ,'Medical_History_34']

data_6= data_5
data_6= data_6.drop(data_6[drop_list_6],axis=1)
mse6 = linearModel(data_6) # mse =0.636
regressor_OLS_6 = lin_regressor_summ(data_6, y)
regressor_OLS_6.summary() # R-sq =0.358

# Linear Model 7#
# use svd to predict
def pred_svd (df):
    col_mean = np.nanmean(df, axis=0, keepdims=1)
    valid = np.isfinite(df)
    df0 = np.where(valid, df, col_mean)
    halt = True
    maxiter =100
    ii = 1
#    normlist = []
    while halt == True:
        U, s, V = np.linalg.svd(df0, full_matrices = False)
        s1 = [(i*0 if i <= 30 else i ) for i in s]
        df1 = U.dot(np.diag(s1).dot(V))
        df2= np.where(~valid, df1, df0)
        norm = np.linalg.norm(df2 - df1)
#        normlist.append(norm)
#        print(norm)
        df0=df2
        if norm < 0.00001 or ii >= maxiter:
            halt = False
#            error = np.nansum((df1-df)**2)
        ii += 1
        print(ii)
    return df1

data_7_df =norm_data
data_7 = data_7_df.iloc[0:59381,:]
y_svd_pred = pred_svd(data_7)
y_svd_pred_df = pd.DataFrame(data= y_svd_pred,columns= norm_data.columns)
y_pred = pd.DataFrame(y_svd_pred_df['Response'])
mse7 = np.sum((y_pred- y)**2)/len(y) #mse =0.0000000731

# Linear Model 8 #
# X: normalised,fill avg,MBI*age
# y: normalised
data_8_df= norm_data_dropResponse
data_8= fill_avg(data_8_df).iloc[0:59381,:]
data_8['BMI_Age']= data_8['BMI']*data_8['Ins_Age']
mse8= linearModel(data_8) # mse = 0.616
regressor_OLS_8 = lin_regressor_summ(data_8, y)
regressor_OLS_8.summary() # R-sq = 0.378