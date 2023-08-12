#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib
from urllib import request
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz", "housing.tgz")


# In[3]:


# Open the tar file and extract its contents
with tarfile.open('housing.tgz', 'r:gz') as tar:
    tar.extractall()

# Read a CSV file inside the extracted contents using pandas
df = pd.read_csv('housing.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


ocean_proximity = df['ocean_proximity']
ocean_proximity.value_counts()


# In[7]:


ocean_proximity.describe()


# In[8]:


df.hist(bins=50, figsize=(20,15))
plt.savefig("obraz1.png")


# In[9]:


df.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1, figsize=(7,4))
plt.savefig("obraz2.png")


# In[10]:


df.plot(kind="scatter", x="longitude", y="latitude",alpha=0.4, figsize=(7,3), colorbar=True,s=df["population"]/100, label="population",c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig("obraz3.png")


# In[11]:


df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={'index' : 'atrybut', 'median_house_value' : 'wspolczynnik_korelacji'}).to_csv('korelacja.csv', index = False)


# In[12]:


sns.pairplot(df)


# In[13]:


train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)
len(train_set),len(test_set)


# In[14]:


train_set.head()


# In[15]:


test_set.head()


# In[16]:


train_set.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={'index' : 'atrybut', 'median_house_value' : 'wspolczynnik_korelacji'})


# In[17]:


test_set.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={'index' : 'atrybut', 'median_house_value' : 'wspolczynnik_korelacji'})


# In[18]:


train_set.to_pickle('train_set.pkl')


# In[19]:


train_set.to_pickle('test_set.pkl')

