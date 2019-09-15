# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:54:09 2019

@author: Dheeraj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


boston_data=load_boston()

df_x=pd.DataFrame(boston_data.data , columns = boston_data.feature_names)
df_y=pd.DataFrame(boston_data.target)

df_x=df_x['RM']

X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, train_size=0.8, random_state=45) 
X_train= X_train[:,np.newaxis]
X_test= X_test[:,np.newaxis]
lr = LinearRegression()
lr = lr.fit(X_train, Y_train)
ypr = lr.predict(X_test)
plt.scatter(X_train, Y_train,c='green')
plt.plot(X_test, lr.predict(X_test))
plt.scatter(X_test,Y_test,c='red')
plt.xlabel("df_x")
plt.ylabel("df_y")
plt.title("SIMPLE LINEAR REGRESSION")
plt.show()
df_y_test=pd.DataFrame(Y_test,ypr)
df_y_pr=pd.DataFrame(ypr)
df_y1=pd.merge(df_y_test,df_y_pr,how='inner')
a=lr.coef_
c=lr.intercept_
r2=r2_score(ypr,Y_test)
 
 