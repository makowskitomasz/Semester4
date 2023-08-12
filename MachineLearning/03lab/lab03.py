#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pickle


# In[4]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[5]:


list_to_df = []
list_of_tuples = []


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[7]:


tmp_list = []
lin_reg = LinearRegression()
lin_reg.fit(X_train.reshape(-1, 1), y_train)
print(lin_reg.intercept_, lin_reg.coef_, "\n")
y_train_pred = [lin_reg.intercept_ * element + lin_reg.coef_ for element in X_train]
y_test_pred = [lin_reg.intercept_ * element + lin_reg.coef_ for element in X_test]
lin_reg_train_mse = mean_squared_error(y_train, y_train_pred)
lin_reg_test_mse = mean_squared_error(y_test, y_test_pred)
tmp_list.append(lin_reg_train_mse)
tmp_list.append(lin_reg_test_mse)
list_to_df.append(tmp_list)
print(tmp_list)
print(list_to_df)


# In[8]:


list_of_tuples.append((lin_reg, None))


# In[9]:


knn_reg_3 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_reg_3.fit(X_train.reshape(-1, 1), y_train)
y_knn3_train_pred = knn_reg_3.predict(X_train.reshape(-1, 1))
y_knn3_test_pred = knn_reg_3.predict(X_test.reshape(-1, 1))
knn_3_train_mse = mean_squared_error(y_train, y_knn3_train_pred)
knn_3_test_mse = mean_squared_error(y_test, y_knn3_test_pred)
tmp_list = []
tmp_list.append(knn_3_train_mse)
tmp_list.append(knn_3_test_mse)
list_to_df.append(tmp_list)


# In[10]:


list_of_tuples.append((knn_reg_3, None))


# In[11]:


knn_reg_5 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_reg_5.fit(X_train.reshape(-1, 1), y_train)
y_knn5_train_pred = knn_reg_5.predict(X_train.reshape(-1, 1))
y_knn5_test_pred = knn_reg_5.predict(X_test.reshape(-1, 1))
knn_5_train_mse = mean_squared_error(y_train, y_knn5_train_pred)
knn_5_test_mse = mean_squared_error(y_test, y_knn5_test_pred)
tmp_list = []
tmp_list.append(knn_5_train_mse)
tmp_list.append(knn_5_test_mse)
list_to_df.append(tmp_list)


# In[12]:


list_of_tuples.append((knn_reg_5, None))


# In[13]:


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
print(lin_reg.intercept_, lin_reg.coef_)
y_poly2_train_pred = lin_reg.predict(poly_features.fit_transform(X_train.reshape(-1, 1)))
y_poly2_test_pred = lin_reg.predict(poly_features.fit_transform(X_test.reshape(-1, 1)))
y_poly2_train_mse = mean_squared_error(y_train, y_poly2_train_pred)
y_poly2_test_mse = mean_squared_error(y_test, y_poly2_test_pred)
tmp_list = []
tmp_list.append(y_poly2_train_mse)
tmp_list.append(y_poly2_test_mse)
list_to_df.append(tmp_list)


# In[14]:


list_of_tuples.append((lin_reg, poly_features))


# In[15]:


poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
print(lin_reg.intercept_, lin_reg.coef_)
y_poly3_train_pred = lin_reg.predict(poly_features.fit_transform(X_train.reshape(-1, 1)))
y_poly3_test_pred = lin_reg.predict(poly_features.fit_transform(X_test.reshape(-1, 1)))
y_poly3_train_mse = mean_squared_error(y_train, y_poly3_train_pred)
y_poly3_test_mse = mean_squared_error(y_test, y_poly3_test_pred)
tmp_list = []
tmp_list.append(y_poly3_train_mse)
tmp_list.append(y_poly3_test_mse)
list_to_df.append(tmp_list)


# In[16]:


list_of_tuples.append((lin_reg, poly_features))


# In[17]:


poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
print(lin_reg.intercept_, lin_reg.coef_)
y_poly4_train_pred = lin_reg.predict(poly_features.fit_transform(X_train.reshape(-1, 1)))
y_poly4_test_pred = lin_reg.predict(poly_features.fit_transform(X_test.reshape(-1, 1)))
y_poly4_train_mse = mean_squared_error(y_train, y_poly4_train_pred)
y_poly4_test_mse = mean_squared_error(y_test, y_poly4_test_pred)
tmp_list = []
tmp_list.append(y_poly4_train_mse)
tmp_list.append(y_poly4_test_mse)
list_to_df.append(tmp_list)


# In[18]:


list_of_tuples.append((lin_reg, poly_features))


# In[19]:


poly_features = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly_features.fit_transform(X_train.reshape(-1, 1))
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
print(lin_reg.intercept_, lin_reg.coef_)
y_poly5_train_pred = lin_reg.predict(poly_features.fit_transform(X_train.reshape(-1, 1)))
y_poly5_test_pred = lin_reg.predict(poly_features.fit_transform(X_test.reshape(-1, 1)))
y_poly5_train_mse = mean_squared_error(y_train, y_poly5_train_pred)
y_poly5_test_mse = mean_squared_error(y_test, y_poly5_test_pred)
tmp_list = []
tmp_list.append(y_poly5_train_mse)
tmp_list.append(y_poly5_test_mse)
list_to_df.append(tmp_list)


# In[20]:


list_of_tuples.append((lin_reg, poly_features))


# In[21]:


for element in list_to_df:
    print(element)


# In[22]:


df = pd.DataFrame(list_to_df, columns=['train_mse', 'test_mse'])
df.index = ['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg']
         


# In[23]:


print(df)


# In[24]:


df.to_pickle('mse.pkl')


# In[25]:


for element in list_of_tuples:
    print(element)


# In[26]:


with open ('reg.pkl', 'wb') as file:
    pickle.dump(list_of_tuples, file)

