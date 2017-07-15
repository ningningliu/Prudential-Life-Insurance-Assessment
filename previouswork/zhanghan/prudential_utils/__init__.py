__author__ == 'zhanghan'

# utils.py:  1. load_data_tree()
# 			> data preprocessing / feature engineering  for tree-based models

# 		   2. data_normalization
# 		   	> normlizing processed data from load_data_tree funciton, results will be using for other non-tree-based models 

# 		   3. eval_wrapper(pred, actual):
# 		   	> return quadratic weighted score for predicitons


# 		   4. modelfit_xgb(model, data, predictor_columns, ... )
# 		   	> cross validation for xgboost model
# 		   	> return optimal number of estimators for particular learning_rate

# 		   5. grid_search_xgb
# 		    > parameters grid search for xgboost and return model with updated params. 

# 		   parmas_list(1 - 7)
# 		    > xgboost parameters to be tuned with cross validation 

# paths.py: files path infomation
