import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from sklearn.model_selection import train_test_split

from scipy.stats import kendalltau, spearmanr

# random forest
# split train data into 2 part
df = pd.read_csv('processed_train.csv')
cols = df.columns.values
X = df[cols[1:-1]]
y = df['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# test data to be predicted
testx = pd.read_csv('processed_test.csv')
cols = testx.columns
testx = testx[cols[1:]]

id = pd.read_csv('test.csv')['Id']
submit = pd.DataFrame(id)


# parameter
PARAMETER_GRID = [
    (100, 200, 300, 400, 500),  # n_estimators
    (0.3, 0.4, 0.5,)  # max_features
]

# set score threshold
best_score = float('-inf')

# random forest modeling
print('\n')
print('start modeling')
for n, f in product(*PARAMETER_GRID):
    est = RandomForestRegressor(oob_score=True,
                                n_estimators=n,
                                max_features=f,
                                n_jobs=-1,  # use all of computer cores
                                min_samples_split=2,  # least samples split condition
                                min_samples_leaf=1,  # least leaf generate condition
                                min_weight_fraction_leaf=0,
                                max_leaf_nodes=None,
                                )
    est.fit(X_train, y_train)
    # model evaluation
    print('n_estimators: ' + str(n))
    print('max_features: ' + str(f))
    buff_y = est.predict(X_test)
    buff_y = np.round(buff_y).astype('int')
    buff_y = pd.Series(buff_y, index=y_test.index)
    #print('R square between predict and actual value y: \n' + str(est.score(X_test, y_test)))
    estscore = kappa(y_test, buff_y)

    print('Kappa evaluation: \n' + str(estscore))
    #buff_r, buff_p = spearmanr(buff_y, y_test)
    #print('spearmanr coefficient: ' + str(buff_r))
    #print('p value: ' + str(buff_p))
    print('\n')
    if estscore > best_score:
        best_score, best_est = estscore, est

# best model for training data
test_target = est.predict(X_test)
test_target = test_target.round().astype('int')
test_target_series = pd.Series(index=y_test.index.values, data=test_target)


# prediction to submit
final_target = est.predict(testx)
final_target = final_target.round().astype('int')
submit['Response'] = final_target
submit.to_csv('rf_predict.csv', index=False)

# feature importance
feature_df = pd.DataFrame(cols, columns=['name'])
feature_df['importance'] = est.feature_importances_
feature_df.sort_values('importance', ascending=False, inplace=True)