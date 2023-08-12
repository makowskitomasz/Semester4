#!/usr/bin/env python
# coding: utf-8

# In[227]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle


# In[228]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True).frame


# In[229]:


data_iris = datasets.load_iris(as_frame=True).frame


# In[230]:


X = data_breast_cancer.drop('target', axis=1)
y = data_breast_cancer['target']


# In[231]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[232]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[233]:


X_train = X_train[['mean area', 'mean smoothness']]
X_test = X_test[['mean area', 'mean smoothness']]


# In[234]:


svm_model1 = Pipeline([('linear_svc', LinearSVC(C=625, loss='hinge', random_state=42))])
svm_model1.fit(X_train, y_train)


# In[235]:


svm_model2 = Pipeline([('scaler', StandardScaler()), ('linear_svc', LinearSVC(loss='hinge', random_state=42))])
svm_model2.fit(X_train, y_train)


# In[236]:


accuracy_list = []


# In[237]:


train_accuracy_svm1 = svm_model1.score(X_train, y_train)
test_accuracy_svm1 = svm_model1.score(X_test, y_test)
train_accuracy_svm2 = svm_model2.score(X_train, y_train)
test_accuracy_svm2 = svm_model2.score(X_test, y_test)


# In[238]:


print(train_accuracy_svm1)
print(test_accuracy_svm1)
print(train_accuracy_svm2)
print(test_accuracy_svm2)


# In[239]:


accuracy_list.append(train_accuracy_svm1)
accuracy_list.append(test_accuracy_svm1)
accuracy_list.append(train_accuracy_svm2)
accuracy_list.append(test_accuracy_svm2)


# In[240]:


print(accuracy_list)


# In[241]:


with open('bc_acc.pkl', 'wb') as file:
    pickle.dump(accuracy_list, file)


# In[242]:


X = data_iris.drop('target', axis=1)
y = (data_iris['target'] == 2).astype(np.int8)
X = X[['petal length (cm)', 'petal width (cm)']]


# In[243]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)


# In[244]:


svm_model1 = Pipeline([('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])
svm_model1.fit(X_train, y_train)


# In[245]:


svm_model2 = Pipeline([('scaler', StandardScaler()), ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])
svm_model2.fit(X_train, y_train)


# In[246]:


train_accuracy_svm1 = svm_model1.score(X_train, y_train)
test_accuracy_svm1 = svm_model1.score(X_test, y_test)
train_accuracy_svm2 = svm_model2.score(X_train, y_train)
test_accuracy_svm2 = svm_model2.score(X_test, y_test)


# In[247]:


print(train_accuracy_svm1)
print(test_accuracy_svm1)
print(train_accuracy_svm2)
print(test_accuracy_svm2)


# In[248]:


accuracy_list = []
accuracy_list.append(train_accuracy_svm1)
accuracy_list.append(test_accuracy_svm1)
accuracy_list.append(train_accuracy_svm2)
accuracy_list.append(test_accuracy_svm2)


# In[249]:


print(accuracy_list)


# In[250]:


with open('iris_acc.pkl', 'wb') as file:
    pickle.dump(accuracy_list, file)

