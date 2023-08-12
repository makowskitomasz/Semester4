#!/usr/bin/env python
# coding: utf-8

# In[55]:


from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle


# In[56]:


data_breast_cancer = datasets.load_breast_cancer()
data_iris = load_iris()


# In[57]:


breast_X = data_breast_cancer.data
breast_y = data_breast_cancer.target


# In[58]:


pca_breast = PCA(n_components=0.9)
breast_reduced = pca_breast.fit_transform(breast_X)
print(breast_X.shape, ' ---> ', breast_reduced.shape)
print(pca_breast.explained_variance_ratio_)


# In[59]:


scaler = StandardScaler()
scaled_breast = pd.DataFrame(scaler.fit_transform(breast_X))
scaled_breast_reduced = pca_breast.fit_transform(scaled_breast)
print(scaled_breast.shape, ' ---> ', scaled_breast_reduced.shape)
print(pca_breast.explained_variance_ratio_)
print(sum(pca_breast.explained_variance_ratio_))


# In[60]:


with open('pca_bc.pkl', 'wb') as filename:
    pickle.dump(pca_breast.explained_variance_ratio_, filename)


# In[61]:


iris_X = data_iris.data
iris_y = data_iris.target


# In[62]:


pca_iris = PCA(n_components=0.9)
iris_reduced = pca_iris.fit_transform(iris_X)
print(iris_X.shape, ' ---> ', iris_reduced.shape)
print(pca_iris.explained_variance_ratio_)


# In[63]:


scaled_iris = pd.DataFrame(scaler.fit_transform(iris_X))
scaled_iris_reduced = pca_iris.fit_transform(scaled_iris)
print(scaled_iris.shape, ' ---> ', scaled_iris_reduced.shape)
print(pca_iris.explained_variance_ratio_)
print(sum(pca_iris.explained_variance_ratio_))


# In[64]:


with open('pca_ir.pkl', 'wb') as filename:
    pickle.dump(pca_iris.explained_variance_ratio_, filename)


# In[70]:


indices_breast = [np.argmax(abs(pca_breast.components_[i])) for i in range(len(pca_breast.components_))]
print(indices_breast)


# In[66]:


with open('idx_bc.pkl', 'wb') as filename:
    pickle.dump(indices_breast, filename)


# In[69]:


indices_iris = [np.argmax(abs(pca_iris.components_[i])) for i in range(len(pca_iris.components_))]
print(indices_iris)


# In[68]:


with open('idx_ir.pkl', 'wb') as filename:
    pickle.dump(indices_iris, filename)

