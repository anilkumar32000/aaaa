# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:27:57 2020

@author: Dell
"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import pickle


data=pd.read_csv('dataset.csv', error_bad_lines=False)
print(data.head())

#normalise data
data=abs((data-data.mean())/data.std())
print(data.head())

#create X and y matrices

X=data.iloc[:,0:4]
X=X.values

y1=data.iloc[:,4:5]
y1=y1.values  #convert y to numpy array


y2=data.iloc[:,5:6]
y2=y2.values  #convert y to numpy array


y3=data.iloc[:,6:7]
y3=y3.values  #convert y to numpy array

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor2 = LinearRegression()
regressor3 = LinearRegression()

# Fitting model with trainig data
a=regressor1.fit(X, y1)
b=regressor2.fit(X, y2)
c=regressor3.fit(X, y3)

# Saving model to disk
pickle.dump(a, open('model1.pkl','wb'))
pickle.dump(b, open('model2.pkl','wb'))
pickle.dump(c, open('model3.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
print(abs(model1.predict([[2000, 4, 4, 6]])))

# Loading model to compare the results
model2 = pickle.load(open('model2.pkl','rb'))
print(abs(model2.predict([[2000, 4, 4, 6]])))

# Loading model to compare the results
model3 = pickle.load(open('model3.pkl','rb'))
print(abs(model3.predict([[2000, 4, 4, 6]])))