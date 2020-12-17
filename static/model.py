# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 07:56:20 2020

@author: Dell
"""
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import pickle

def get_powers(degree):
    l=[0,1,2,3,4,5]
    powers=[]
    for i in range(1,degree+1):
        powers.append([x for x in combinations_with_replacement(l,i)])
    powers_flattened=[]
    for sublist in powers:
        for x in sublist:
            powers_flattened.append(x)
    return powers_flattened


def transform_data(X,powers):
    X_new=np.ones((X.shape[0],len(powers)))
    for n in range(X.shape[0]):
        #print(n)
        for i in range(len(powers)):
            for j in powers[i]:
                X_new[n][i]=X_new[n][i]*X[n][j]
    return X_new


data=pd.read_csv('data.csv', error_bad_lines=False)
print(data.head())

#normalise data
data=abs((data-data.mean())/data.std())
print(data.head())

#create X and y matrices

X=data.iloc[:,0:6]
X=X.values

y1=data.iloc[:,6:7]
y1=y1.values  #convert y to numpy array

powers_5=get_powers(5)
X_5=transform_data(X,powers_5)
X_5=(X_5-X_5.mean())/X_5.std()
np.save('X_5.npy',X_5)

# Splitting Training and Test Set
# Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X, y1)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2000, 9, 6, 4, 6, 8]]))
# # Fitting Linear Regression to the dataset
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()

# # Fitting Polynomial Regression to the dataset
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=5)
# X_poly = poly_reg.fit_transform(X_5)
# pol_reg = LinearRegression()
# pol_reg.fit(X_poly, y)

# # Saving model to disk
# pickle.dump(poly_reg, open('model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))


