import random
import numpy as np
from ml_metrics import quadratic_weighted_kappa
from scipy.optimize import fmin_powell
from sklearn.cross_validation import  StratifiedKFold

num_classes = 8

# regress_model = joblib.load('models/regress.pkl')
# rf_model =joblib.load('models/rf.pkl')
# svm_model = joblib.load('models/svm.pkl')
# xgb_model = joblib.load('models/xgb.pkl')

random.seed(26)


# train = pd.read_csv(path.train)
# test = pd.read_csv(path.test)

# train_normal = pd.read_csv(path.train_normal)
# test_normal = pd.read_csv(path.test_normal)

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

# class modelOffset():
#     def __init__(self, model, X, y, random_seed=None, shuffle=False):
#         self.model = model
#         self.random_seed = 26
#         self.shuffle = shuffle
#         self.X = X
#         self.y = y
#         if self.shuffle:
#             idx = np.random.permutation(y.size)
#             self.X = self.X.iloc[idx]
#             self.y = self.y.iloc[idx]


#     def one_fold_train(self, X_train, y_train, X_test, y_test):
#         self.model.fit(X_train, y_train)
#         train_preds = self.model.predict(X_train)
#         test_preds = self.model.predict(X_test)

#         train_preds = np.clip(train_preds, 1, 8)
#         test_preds = np.clip(test_preds, 1, 8)

#         offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])

#         offset_preds = np.vstack((train_preds, train_preds, y_train))
#         offset_preds = apply_offset(offset_preds, offsets)

#         opt_order = list(range(8))
#         for j in opt_order:
#             train_offset = lambda x: -score_offset(offset_preds, x, j) * 100
#             offsets[j] = fmin_powell(train_offset, offsets[j], disp=True)

#         test_offset = np.vstack((test_preds, test_preds))
#         test_offset = apply_offset(test_offset, offsets)
#         final_test_preds = np.round(np.clip(test_offset[1], 1, 8)).astype(int)

#         return final_test_preds


#     def offsetResult(self, n_folds=5):
#         skf = list(StratifiedKFold(self.y, n_folds))
#         preds_list = []
#         for i, (train, test) in enumerate(skf):
#             print(str(i) + " th fold")
#             X_train = self.X.iloc[train]
#             y_train = self.y.iloc[train]
#             X_test = self.X.iloc[test]
#             y_test = self.y.iloc[test]
#             final_test_preds = self.one_fold_train(X_train, y_train, X_test, y_test)
#             preds_list.append(final_test_preds)

#         denested_preds_list = [inner for outer in preds_list for inner in outer]

#         return denested_preds_list

def offset_apply(train_preds, test_preds, y_train):

	train_preds = np.clip(train_preds, -0.99, 8.99)
	test_preds = np.clip(test_preds, -0.99, 8.99)

	offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])

	offset_preds = np.vstack((train_preds, train_preds, y_train))
	offset_preds = apply_offset(offset_preds, offsets)

	opt_order = list(range(8))
	for j in opt_order:
		train_offset = lambda x: -score_offset(offset_preds, x, j) * 100
		offsets[j] = fmin_powell(train_offset, offsets[j], disp=True)

	train_offset_out = np.vstack((train_preds, train_preds))
	train_offset_out = apply_offset((train_preds, offsets))
	final_train_preds = np.round(np.clip(train_offset_out[1], 1, 8)).astype(int)
	test_offset = np.vstack((test_preds, test_preds))
	test_offset = apply_offset(test_offset, offsets)
	final_test_preds = np.round(np.clip(test_offset[1], 1, 8)).astype(int)

	return final_train_preds, final_test_preds 






