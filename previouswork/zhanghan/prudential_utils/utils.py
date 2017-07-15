import pandas as pd
from copy import deepcopy
import numpy as np
from local import paths
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import operator
import random


random.seed(26)
print('Load the data using pandas')

path_train = paths.TRAIN_PATH
path_test = paths.TEST_PATH
path_submission = paths.SUBMISSION_PATH

def load_data_tree():
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    all_data = train.append(test)

    # fill test response with -1
    all_data['Response'].fillna(-1, inplace=True)
    all_data_non_response = deepcopy(all_data)
    all_data_non_response.drop(['Response'], axis=1, inplace=True)
    # feature engineering
    # 1.NaN values counting
    # 2.BMI age interaction
    # 3.Product_Info_2 factorize
    # 4.Medical History values summing up

    NA_count_list = []
    for row in range(len(all_data_non_response)):
        NA_count_list.append(all_data_non_response.iloc[row].isnull().sum())
    all_data['NA_count'] = NA_count_list

    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    Employ_keyword_columns = all_data.columns[all_data.columns.str.startswith('Employment_Info_')]
    for employ in Employ_keyword_columns:
        all_data[employ].fillna(-1, inplace=True)


    Family_keyword_columns = all_data.columns[all_data.columns.str.startswith('Family_Hist_')]
    for family in Family_keyword_columns:
        all_data[family].fillna(all_data[family].mean(), inplace=True)

    all_data['Insurance_History_5'].fillna(all_data['Insurance_History_5'].mean(), inplace=True)
    all_data['Medical_History_1'].fillna(all_data['Medical_History_1'].median(), inplace=True)
    # important feature
    all_data['Medical_History_15'].fillna(float(0), inplace=True)
    all_data['Response'] = all_data['Response'].astype(int)

    Medical_history_columns = all_data.columns[all_data.columns.str.startswith('Medical_History_')]
    for medical in Medical_history_columns:
        all_data[medical].fillna(-1, inplace=True)

    binarizer = preprocessing.Binarizer()


    train_processed_temp = all_data[all_data['Response'] > 0].copy()
    target = train_processed_temp['Response']
    test_processed_tree = all_data[all_data['Response'] < 0].copy()
    test_processed_tree = test_processed_tree.drop('Response', axis=1)

    cols = list(train_processed_temp.columns.values)
    cols.remove('Id')
    corcoef = {}
    for col in cols:
        cor = np.corrcoef(train_processed_temp[col], train_processed_temp['Response'])
        corcoef[col] = cor[0, 1]
    sorted_cor = sorted(corcoef.items(), key=operator.itemgetter(1))

    # most 5 correlative
    positive_correlation = sorted_cor[-5:]

    variable_list = ['Medical_History_23', 'Product_Info_4', 'Medical_History_39','Medical_History_4','Medical_History_23']
    quad_list = [item+str('_quad') for item in variable_list]
    for i in range(len(variable_list)):
        all_data[quad_list[i]] = all_data[variable_list[i]] ** 2


    """
    polynomial feature particularly for medical_keyword
    -> 140 columns
    """

    poly_list = ['Medical_Keyword_15', 'Medical_Keyword_3', 'Med_Keywords_Count']
    poly = PolynomialFeatures(degree=3, interaction_only=True)
    poly.fit_transform(all_data[poly_list])

    all_data['Medical_Keyword_15_3'] = all_data['Medical_Keyword_15'] * all_data['Medical_Keyword_3']
    all_data['Medical_Keyword_3_Count'] = all_data['Medical_Keyword_3'] * all_data['Med_Keywords_Count']
    all_data['Medical_Keyword_15_Count'] = all_data['Medical_Keyword_15'] * all_data['Med_Keywords_Count']
    all_data['Medical_Keyword_3_15_Count'] = all_data['Medical_Keyword_3'] * all_data['Medical_Keyword_15'] * all_data['Med_Keywords_Count']
    all_data['Response'] = all_data['Response'].astype(int)

    train_processed_tree = all_data[all_data['Response'] > 0].copy()
    target = train_processed_tree['Response']
    test_processed_tree = all_data[all_data['Response'] < 0].copy()
    test_processed_tree = test_processed_tree.drop('Response', axis=1)

    return train_processed_tree, test_processed_tree

def data_normalization(train, test):

    all_data = train.append(test)
    all_data['Response'].fillna(-1, inplace=True)
    all_data_no_response = deepcopy(all_data)
    all_data_no_response.drop('Response', axis=1, inplace=True)

    # normalization

    cols_normalized = list(all_data_no_response.columns.values)
    cols_normalized.remove("Id")
    scalar = preprocessing.StandardScaler()
    all_data_no_response[cols_normalized] = scalar.fit_transform(all_data_no_response[cols_normalized])

    all_data_no_response['Response'] = all_data['Response'].astype(int)
    train_processed_linear = all_data_no_response[all_data_no_response['Response'] > 0].copy()
    test_processed_linear = all_data_no_response[all_data_no_response['Response'] < 0].copy()
    test_processed_linear = test_processed_linear.drop('Response', axis=1)

    return train_processed_linear, test_processed_linear





if __name__ =='__main__':
    train_tree, test_tree = load_data_tree()

    train_tree.to_csv('data/train_tree.csv', index=False)
    test_tree.to_csv('data/test_tree.csv', index=False)

    train_regress, test_regress = data_normalization(train_tree, test_tree)
    train_regress.to_csv('data/train_regress.csv', index=False)
    test_regress.to_csv('data/test_regress.csv', index=False)














