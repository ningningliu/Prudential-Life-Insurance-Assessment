import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn import preprocessing
import operator
import random
import path
from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

random.seed(26)
print('Load the data')
path_train = path.TRAIN_PATH
path_test = path.TEST_PATH


def fill_svd(df):
	col_mean = np.nanmean(df, axis=0, keepdims=1)
	valid = np.isfinite(df)
	df0 = np.where(valid, df, col_mean)
	halt = True
	maxiter = 100
	ii = 1
	normlist = []
	while halt == True:
	    U, s, V = np.linalg.svd(df0, full_matrices=False)
	    s1 = [(i * 0 if i <= 30 else i) for i in s]
	    df1 = U.dot(np.diag(s1).dot(V))
	    df2 = np.where(~valid, df1, df0)
	    norm = np.linalg.norm(df2 - df1)
	    normlist.append(norm)
	    #        print(norm)
	    df0 = df2
	    if norm < 0.00001 or ii >= maxiter:
	        halt = False
	        error = np.nansum((df1 - df) ** 2)
	    ii += 1
	print(ii)
	return df2, normlist, error

def loadParseData(imputing='mean'):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    all_data = train.append(test)

    # fill test response with -1
    all_data['Response'].fillna(-1, inplace=True)
    all_data_replicate = deepcopy(all_data)
    all_data.drop(['Response'], axis=1, inplace=True)
    print('\n')
    print('Normal feature engineering\n \
    	1. NaN values count up \n \
    	2. BMI age interaction \n \
    	3. Product_Info_2 factorize \n \
    	4. Medical History values sum up')

    print('NaN values counting \n')
    NA_count_list = []
    for row in range(len(all_data)):
    	NA_count_list.append(all_data.iloc[row].isnull().sum())
    all_data['NA_count'] = NA_count_list

    print('BMI * Age interaction term \n')

    all_data["BMI_Ins_age"] = all_data.BMI * all_data.Ins_Age
    

    print('product_Info_2 factorizing \n')
    all_data["Product_Info_2_char"] = all_data.Product_Info_2.str[0]
    all_data["Product_Info_2_num"] = all_data.Product_Info_2.str[1]

    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]


    print('medical keyworkds counting')
    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]

    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    # svd branch
    if imputing == 'svd':
    	print('start svd iteration for verifying missing values')
    	print('max iteration = 100 \n ')
    	predictors = [col for col in all_data.columns.values if col not in ['Response', 'Id']]
    	svd_df, normlist, error = fill_svd(all_data[predictors])
    	all_data = svd_df
    	all_data['Response'] = all_data_replicate['Response']	
    	all_data['Id'] = all_data_replicate['Id']
    	print('(. \n)*10')
    	print('finshied, return parsed data with svd method')
    	return all_data

    # mean method for imputing missing values

    print('general imputing for missing values either mean or median or -1')
    Emplot_keyword_columns = all_data.columns[all_data.columns.str.startswith('Employment_Info_')]
    for employ in Emplot_keyword_columns:
        all_data[employ].fillna(-1, inplace=True)

    Family_keyword_columns = all_data.columns[all_data.columns.str.startswith('Family_Hist_')]
    for family in Family_keyword_columns:
        all_data[family].fillna(all_data[family].mean(), inplace=True)

    all_data['Insurance_History_5'].fillna(all_data['Insurance_History_5'].mean(), inplace=True)
    all_data['Medical_History_1'].fillna(all_data['Medical_History_1'].median(), inplace=True)

    # important feature
    all_data['Medical_History_15'].fillna(float(0), inplace=True)
    all_data['Response'] = all_data_replicate['Response'].astype(int)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    col = all_data.columns.values
    all_data = pd.DataFrame(imp.fit_transform(all_data))
    all_data.columns = col
    print('return parsed training and testing data with mean method imputing')
    train_data = all_data[all_data['Response'] > 0]
    test_data = all_data[all_data['Response'] < 0]
    test_data.drop('Response',axis=1, inplace=True)
    return train_data, test_data

# feature selection by extra trees
# number of trees = 300, importances threshold = 0.005
# provide alternative feature selecting scheme
# 1. select those features who cumulate to 90% importance of all features
# 2. select those features who has more than 0.5% importances

def normalization(train, test):
	all_data = train.append(test)
	all_data['Response'].fillna(-1, inplace=True)
	all_data_no_response = deepcopy(all_data)
	all_data_no_response.drop('Response', axis=1, inplace=True)

	# normalization with minmax  scaler method
	print('standardize features by removing the mean and scaling to unit variance')
	cols_normalized = list(all_data_no_response.columns.values)
	cols_normalized.remove("Id")
	scalar = preprocessing.MinMaxScaler()
	all_data_no_response[cols_normalized] = scalar.fit_transform(all_data_no_response[cols_normalized])
	all_data_no_response['Response'] = all_data['Response'].astype(int)

	train_normalized = all_data_no_response[all_data_no_response['Response'] > 0 ].copy()
	test_normalized = all_data_no_response[all_data_no_response['Response'] < 0].copy()
	test_normalized = test_normalized.drop('Response', axis=1)

	return train_normalized, test_normalized


def et_selection(df, alternatives=True, default_import=0.9):
	if len(df) > 60000:
		return('wrong data frame input, specify the training section')
	if max(df['Product_Info_3']) < 10 :
		return('wrong data frame input, specify the normalized data')

	predictors = [col for col in df.columns.values if col not in ['Response', 'Id']]
	extra_model = ExtraTreesClassifier(n_estimators=300, random_state=26)
	extra_model.fit(df[predictors], df['Response'])
	# importances_df = pd.DataFrame({'features': predictors,
    #                                'importances': extra_model.feature_importances_})

	extra_weights = pd.DataFrame(extra_model.feature_importances_, columns=['importances'], index=predictors)
	sorted_weights = extra_weights.sort_values(by='importances', ascending=False)
	weights_cum = np.cumsum(sorted_weights)
    # plot to be add ...
    # alternative 1 : select features for importances sum up > 0.9
	if alternatives != True:
	    for i in range(len(weights_cum)):
	        if weights_cum.iloc[i].values > default_import:
	        	extra_feature = weights_cum.index[i:]
	        	extra_feature_list = extra_feature.tolist()
	        	print(extra_feature_list)
	        	return extra_feature_list
    # alternative 2: feature select method with SelectFromModel
    # with threshold 0.005
	selected_model = SelectFromModel(extra_model, prefit=True, threshold=0.005)
	extra_feature_array = np.asarray(predictors)[selected_model.get_support().tolist()]
	extra_feature_list = extra_feature_array.tolist()
	print(extra_feature_list)
	return extra_feature_list

def svc_selection(df, p='l1'):
	if len(df) > 60000:
		raise ValueError('wrong data frame input, specify the training section')
	if max(df['Product_Info_3']) > 2:
		raise ValueError('wrong data frame input, specify the normalized data')

	predictors = [col for col in df.columns.values if col not in ['Response', 'Id']]
	if p == 'l1':
		svc_l1 = LinearSVC(C=0.01, penalty='l1', dual=False).fit(df[predictors], df['Response'])
		model_l1 = SelectFromModel(svc_l1, prefit=True)
		l1_feature_list = np.asarray(predictors)[model_l1.get_support().tolist()]
		print(l1_feature_list)
		return l1_feature_list

	elif p == 'l2':
		svc_l2 = LinearSVC(C=0.01, penalty='l2', dual=False).fit(df[predictors])
		model_l2 = SelectFromModel(svc_l2, prefit=True)
		l2_feature_list = np.asarray(predictors)[model_l2.get_support().tolist()]
		return l2_feature_list
	else:
		raise ValueError('Not a valid penalty term, please specify l1 or l2')

