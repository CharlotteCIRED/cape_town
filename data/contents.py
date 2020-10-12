# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:31:33 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import copy

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
from regressors import stats   

contents_value = pd.read_excel('./2. Data/housing_contents.xlsx', sheet_name = "contents", header = 6).iloc[77, 3:11]
dwellings_type = np.transpose(pd.read_excel('./2. Data/housing_contents.xlsx', sheet_name = "dwellings", header = 6).iloc[27:30, 4:12])
dwellings_type.columns = ['Formal', 'Backyard', 'Informal']
disposable_income = pd.read_excel('./2. Data/housing_contents.xlsx', sheet_name = "contents", header = 6).iloc[39, 3:11]
current_income = pd.read_excel('./2. Data/housing_contents.xlsx', sheet_name = "contents", header = 6).iloc[40, 3:11]

Y = copy.deepcopy(dwellings_type)
#Y["disposable_income"] = disposable_income
Y["current_income"] = current_income

reg = LinearRegression(fit_intercept = False).fit(Y, contents_value)
reg.coef_
reg.score(Y, contents_value)

reg = LinearRegression(fit_intercept = False).fit(dwellings_type, contents_value)
reg.coef_
reg.score(dwellings_type, contents_value)


reg = LinearRegression(fit_intercept = False).fit(Y[['current_income']], contents_value)
reg.coef_
reg.score(Y[['current_income']], contents_value)

lin = Lasso(alpha=0.0000000000001,precompute=True,max_iter=10000,
            positive=True, random_state=9999, selection='random', fit_intercept = False)
lin.fit(dwellings_type, contents_value)
lin.coef_ 
stats.coef_pval(rr_scaled, X_train, Y_train)

# Define the Model
model = lambda b, X: b[0] * X.iloc[:, 0] + b[1] * X.iloc[:, 1] + b[2] * X.iloc[:, 2]
model = lambda b, X: b[0] * X.iloc[:, 0] + b[1] * X.iloc[:, 1] + b[2] * X.iloc[:, 2] + b[3] * X.iloc[:, 3]
model = lambda b, X: b[0] * X.iloc[:, 0]  * X.iloc[:, 3] + b[1] * X.iloc[:, 1]  * X.iloc[:, 3] + b[2] * X.iloc[:, 2]
obj = lambda b, Y, X: np.sum(np.abs(Y-model(b, X))**2)
bnds = [(0, None), (0, None), (0, None), (0, None)]
xinit = np.array([0, 0, 0, 0])
res = minimize(obj, args=(contents_value, dwellings_type), x0=xinit, bounds = bnds)
print(f"b1={res.x[0]}, b2={res.x[1]}, b3={res.x[2]}")

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

est = sm.OLS(contents_value, dwellings_type)
est2 = est.fit()
print(est2.summary())