

from ml_metrics import quadratic_weighted_kappa
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.cross_validation import  StratifiedKFold
from scipy.optimize import fmin_powell
from sklearn.metrics.scorer import make_scorer
from local import paths

# ensemble method for stacking five regression models
# stacking
# randomfores : bagging
# xgboost : boosting
# svm
# linear regression (l1 norm, l2 norm)
# 4 models to stack the genrealized model


linear_model = joblib.load('models/linear.pkl')
rf_model =joblib.load('models/rf.pkl')
svm_model = joblib.load('models/svm.pkl')
xgb_model = joblib.load('models/xgb.pkl')

np.random.seed(26)

train_tree = pd.read_csv(paths.TRAIN_TREE)
test_tree = pd.read_csv(paths.TEST_TREE)

train_regress = pd.read_csv(paths.TRAIN_REGRESS)
test_regress = pd.read_csv(paths.TEST_REGRESS)

# variable expect 'Id' and 'Response'
predictors = [col for col in train_regress.columns.values if col not in ['Id','Response']]

# customed evaluation function
myscorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)
num_classes = 8


# helper functions


def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)


def score_offset(data, bin_offset, sv, scorer=eval_wrapper):
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int) == sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

def apply_offset(data, offsets):
    for j in range(num_classes):
        data[1, data[0].astype(int) == j] = data[0, data[0].astype(int)==j] + offsets[j]
    return data

class modelOffset():
    def __init__(self, model, X, y, random_seed=None, shuffle=False):
        self.model = model
        self.random_seed = 26
        self.shuffle = shuffle
        self.X = X
        self.y = y
        if self.shuffle:
            idx = np.random.permutation(y.size)
            self.X = self.X.iloc[idx]
            self.y = self.y.iloc[idx]


    def one_fold_train(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        train_preds = np.clip(train_preds, 1, 8)
        test_preds = np.clip(test_preds, 1, 8)

        offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])

        offset_preds = np.vstack((train_preds, train_preds, y_train))
        offset_preds = apply_offset(offset_preds, offsets)

        opt_order = list(range(8))
        for j in opt_order:
            train_offset = lambda x: -score_offset(offset_preds, x, j) * 100
            offsets[j] = fmin_powell(train_offset, offsets[j], disp=True)

        test_offset = np.vstack((test_preds, test_preds))
        test_offset = apply_offset(test_offset, offsets)
        final_test_preds = np.round(np.clip(test_offset[1], 1, 8)).astype(int)

        return final_test_preds


    def offsetResult(self, n_folds=5):
        skf = list(StratifiedKFold(self.y, n_folds))
        preds_list = []
        for i, (train, test) in enumerate(skf):
            print(str(i) + " th fold")
            X_train = self.X.iloc[train]
            y_train = self.y.iloc[train]
            X_test = self.X.iloc[test]
            y_test = self.y.iloc[test]
            final_test_preds = self.one_fold_train(X_train, y_train, X_test, y_test)
            preds_list.append(final_test_preds)

        denested_preds_list = [inner for outer in preds_list for inner in outer]

        return denested_preds_list



linear_offset = modelOffset(linear_model, train_regress[predictors], train_regress['Response'], shuffle=False)
linear_preds_lists = linear_offset.offsetResult()

rf_offset = modelOffset(rf_model, train_tree[predictors], train_tree['Response'], shuffle=False, random_seed=27)
rf_preds_lists = rf_offset.offsetResult()

svm_offset = modelOffset(svm_model, train_regress[predictors], train_regress['Response'], shuffle=False, random_seed=28)
svm_preds_lists = svm_offset.offsetResult()

xgb_offset = modelOffset(xgb_model, train_tree[predictors], train_tree['Response'], shuffle=False, random_seed=29)
xgb_preds_lists = xgb_offset.offsetResult()


# version 1
# voting ensemble

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
eclf = VotingClassifier(estimators=[('linear',linear_model), ('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)])

train_regress_feature, val_regress_feature, train_regress_target, val_regress_target = train_test_split(train_regress[predictors], train_regress['Response'], test_size=0.2, random_state=26)
train_tree_feature, val_tree_feature, train_tree_target, val_tree_target = train_test_split(train_tree[predictors], train_tree['Response'], test_size=0.2, random_state=26)


linear_model = linear_model.fit(train_regress_feature, train_regress_target)
rf_model = rf_model.fit(train_tree_feature, train_tree_target)
svm_model = svm_model.fit(train_regress_feature, train_regress_target)
xgb_model = xgb_model.fit(train_tree_feature, train_tree_target)

eclf.fit(train_regress_feature, train_regress_target)

e_preds = eclf.predict(val_tree_feature)
e_score = quadratic_weighted_kappa(e_preds, val_regress_target)



# version 2 to do ...
# second layer model

# weighted average
# stacking
# blending
#
from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()
data_dict = {'linear': linear_preds_lists, 'rf': rf_preds_lists, 'svm': svm_preds_lists, 'xgb':xgb_preds_lists}
ensemble_df = pd.DataFrame(data=data_dict, columns=['linear', 'rf', 'svm', 'xgb'])
logis.fit(data_dict['linear', 'rf', 'svm', 'xgb'], train_regress['Response'])

linear_test_preds = linear_offset.one_fold_train(train_regress[predictors], train_regress['Response'], test_regress[predictors])
rf_test_preds = rf_offset.one_fold_train(train_tree[predictors], train_tree['Response'], test_regress[predictors])
svm_test_preds = svm_offset.one_fold_train(train_regress[predictors], train_regress['Response'], test_regress[predictors])
xgb_test_preds = xgb_offset.one_fold_train(train_tree[predictors], train_tree['Response'], test_tree[predictors])

test_dict = {'linear': linear_test_preds, 'rf': rf_test_preds, 'svm': svm_test_preds, 'xgb': xgb_test_preds}
test_df = pd.DataFrame(data=test_dict, columns=['linear', 'rf', 'svm', 'xgb'])


logis_preds = logis.predict(test_df)
logis_preds.to_csv('ensemble_res.csv', index=test_regress.index)


