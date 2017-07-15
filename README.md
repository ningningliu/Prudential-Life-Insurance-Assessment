#### COMP 6208 Advanced Machine Learning : 
##### Group Project Briefing
* Project Team : DataMiner 
* Team members: **Fan Bi**, **Ruiqi Guo**, **Xin Zhang**, **Han Zhang**, **Ningning Liu**
* Project Title: Prudential Life Insurance Assessment

This project aims to develop a predictive model that can classifies risk levels using data points in the existing life insurance assessment. By successful implementation of various advanced machine learning approaches, the results will help Prudential to better understand and quickly to get a quote for new and existing customers, enabling the corporation to make better decision. 

The raw data set is total 26.6 MB in csv format including 20 MB training data. There are over hundred normalised variables describing the existing clients’ attributes e.g. user Id, general health condition, product information, life insurance information, family history and medical background. The target is an ordinal measure of risks with 8 levels relating to final decision associated the applicant. In particular, only 13 out of 127 variables appear to be continuous data whereas the rest is categorical. 

The project is expected to follow the subsequent stages accordingly. Firstly, pre-processing the raw data is essential, particularly when we have significant proportion of missing data in several variables. Next stage is dimensionality reduction over hundred variables. This can be done by feature extraction. Most importantly, we will focus on model building and sustainable optimisation. The predicted target using test set will be assessed by quadratic weighted kappa that measures the agreement between 8 possible risk rating. Python will be used in this project in conjunction with several packages, e.g. pandas, Matplotlib, numpy, scikit-learn and XGBoost. The version control and merging files will be fulfilled by GitHub.

In conclusion, we intend to become acquainted with project pipeline in machine learning through a realistic application in the industry. We will intensively use automated algorithms to continuously optimise the prediction model as well as compare both of excellence and drawbacks during the exploration in each state-of-the-art approach. 
 


* [example](https://github.com/Entheos1994/adml_dataminer/blob/master/example.ipynb)
* [paper](https://github.com/Entheos1994/adml_dataminer/blob/master/project_final.pdf)


* #### Dependencies
-------------------
```
 numpy==1.12.1

 ml_metrics==0.1.4

 pandas==0.19.2

 scipy==0.19.0

 matplotlib==2.0.0

 scikit_learn==0.18.1

 statsmodels==0.8.0

 xgboost==0.6
```


* #### 分享
- 任何想法分享到 [google drive](https://drive.google.com/drive/folders/0B8zqrhAmm5-1VGpSWkN6VW9lc00)

 







