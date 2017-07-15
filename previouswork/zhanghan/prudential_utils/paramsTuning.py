import pandas as pd
import numpy as np
from local import paths
from ml_metrics import quadratic_weighted_kappa
import matplotlib.pyplot as plt
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
import operator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.externals import joblib

np.random.seed(26)

train_tree = pd.read_csv(paths.TRAIN_TREE)
test_tree = pd.read_csv(paths.TEST_TREE)

train_regress = pd.read_csv(paths.TRAIN_REGRESS)
test_regress = pd.read_csv(paths.TEST_REGRESS)

# variable expect 'Id' and 'Response'
predictors = [col for col in train_regress.columns.values if col not in ['Id','Response']]

# customed evaluation function
myscorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)


# def eval_wrapper(yhat, y):
#     y = np.array(y)
#     y = y.astype(int)
#     yhat = np.array(yhat)
#     yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
#     return quadratic_weighted_kappa(yhat, y)


xgb_params_test1 = {
    "max_depth":[3,5,7,9],
    "min_child_weight":[1,3,5]
}

xgb_params_test2 = {
    "max_depth":[4,5,6],
    "min_child_weight":[4,5,6]
}
# tune gamma
xgb_params_test3 = {
    "gamma" :[i/10.0 for i in range(0,5)]
}

# tune subsample and colsample_bytree
xgb_params_test4 = {
    'subsample' : [i/10.0 for i in range(6,10)],
    'colsample_bytree' : [i/10.0 for i in range(6,10)]
}

xgb_params_test5 = {
    'subsample' : [i/100.0 for i in range(75, 90, 5)],
    'colsample_bytree': [i/100.0 for i in range(75,90,5)]
}

# tune regularization parameters
xgb_params_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

xgb_params_test7 = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}

xgb_params_list = [xgb_params_test1, xgb_params_test2, xgb_params_test3, xgb_params_test4, xgb_params_test5, xgb_params_test6, xgb_params_test7]

"""
random forest parameter tuning
n_estimators
criterion
max_depth
min_sample_split
min_sample_leaf
max_feature
"""

rf_params_test1 = {'criterion':['mse','mae']}

rf_params_test2 = {'max_depth':[i for i in range(10,50,5)]}

rf_params_test3 = {'min_samples_split': [2,3,4,5], 'min_samples_leaf':[1,2, 10, 100]}

rf_params_test4 = {'max_features': ['log2','sqrt', 0.2]}

rf_params_test5 = {'max_features': [i/100.0 for i in range(5, 15)]}

rf_params_list = [rf_params_test1, rf_params_test2, rf_params_test3, rf_params_test4, rf_params_test5]


"""
svm parameter tuning
C
gamma
kernel
"""

svm_params_test1 = {'kernel': ['linear', 'poly','rbf', 'sigmoid']}

svm_params_test2 = {'kernel':['poly'], 'degree':[i for i in range(1,6)]}

svm_params_test3 = {'C':[0.01, 0.1, 1.0, 10],
                   'gamma':['auto']}

svm_params_test4 ={'C':[1.0],
                  'gamma':[2**k for k in range(-2,3,1)]}

svm_params_list = [svm_params_test1, svm_params_test2, svm_params_test3, svm_params_test4]




class model_tunning_params():

    # xgboost regressor
    def __init__(self, model_name, random_seed = None, params_list=None):
        self.model_name = model_name
        if model_name == 'xgb':
            self.model = xgb.XGBRegressor(
                                        learning_rate = 0.1,
                                        n_estimators =1000,
                                        max_depth=7,
                                        min_child_weight=1,
                                        gamma=0,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        objective='reg:linear',
                                        nthread=4,
                                        scale_pos_weight=1,
                                        seed=26)
        elif model_name == 'rf':
            self.model = RandomForestRegressor(
                                        n_estimators=100,
                                        criterion='mse',
                                        max_features='sqrt',
                                        max_depth=None,
                                        n_jobs=-1,
                                        verbose=3,
                                        random_state=26
            )
        elif model_name == 'linear':
            self.model = LinearRegression(n_jobs=-1)

        elif model_name =='svm':
            self.model = svm.SVR(C=1.0,
                                 kernel='rbf',
                                 gamma='auto',
                                 verbose=3
                                 )
        else:
            print("not a valid model to tunning parameter \nplease try one of the followings: \n" + '-'*20 + "\n linear \n svm \n xgb \n rf \n" + '-'*20)

        self.params_list = params_list
        self.random_seed = np.random.seed(26)


    # search for best n_estimators and return the updated model
    def modelfit_xgb(self, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, metric='rmse',
                     obt='reg:linear'):
        target = "Response"
        if useTrainCV:
            xgb_param = self.model.get_xgb_params()
            xgb_param['objective'] = obt
            if xgb_param['objective'] == 'multi:softmax':
                xgb_param['num_class'] = 8
                metric = 'merror'
                xgtrain = xgb.DMatrix(dtrain[predictors].values, label=(dtrain[target] - 1).values)
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.model.get_params()['n_estimators'], nfold=cv_folds,
                              metrics=metric, early_stopping_rounds=early_stopping_rounds, verbose_eval=3)
            self.model.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        self.model.fit(dtrain[predictors], dtrain[target], eval_metric=metric)

        # Predict training set:
        dtrain_predictions = self.model.predict(dtrain[predictors])

        if self.model._estimator_type == 'regressor':
            dtrain_prediction = np.clip(dtrain_predictions, 1, 8)
            dtrain_predictions = np.round(dtrain_prediction).astype(int)

        # print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values,
                                                         dtrain_predictions))

        importance = self.model.booster().get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        # plt.figure()
        df = pd.DataFrame(importance, columns=['feature', 'score'])
        df['score'] = df['score'] / df['score'].sum()
        # df.plot()
        df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('importance value')
        # return model which has optimal n_estimators for a specific learning_rate

        return self.model


    def grid_search(self, data, predictors, param_test):
        print('Grid search for '+self.model_name)
        print('Parameter going to be tuned for '+ param_test)
        print('Could be take a long time which depending on your machine resources.')
        while True:
            user_enter = input('Continue:[y/n]')
            if user_enter =='y':
                break
            elif user_enter =='n':
                return ('exit parameter grid search for '+self.model_name+' model')
            else:
                print('not a valid input, please enter [y/n]')

        target = 'Response'
        gsearch = GridSearchCV(estimator=self.model, param_grid=param_test,
                                iid=False, cv=5, scoring=myscorer, n_jobs=-1,
                                verbose=3
                                )

        gsearch.fit(data[predictors], data[target])
        print('\n grid_scores: ', gsearch.grid_scores_)
        print('\n best parameters: ', gsearch.best_params_)
        print('\n best score: ', gsearch.best_score_)

        # update parameters
        for index, value in gsearch.best_params_.items():
            self.model.set_params(**{index: value})

        return self.model, gsearch


    def model_store(self):
        print('store updated model for reporducible result')
        joblib.dump(self.model, 'models/%s.pkl'% (self.model_name))
        print('-----finished-----')




