import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import pickle

data=pd.read_csv('data_1.csv', error_bad_lines=False)
print(data.head())

#normalise data
data=abs((data-data.mean())/data.std())
print(data.head())

#create X and y matrices

X=data.iloc[:,0:4]
X=X.values

y=data.iloc[:,5:6]
y=y.values  #convert y to numpy array

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model2.pkl','wb'))

# Loading model to compare the results
model2 = pickle.load(open('model2.pkl','rb'))
print(abs(model2.predict([[2000, 4, 4, 6]])))