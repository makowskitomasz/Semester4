#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import re


# In[25]:


filename = 'proj2_data.csv'
df = pd.read_csv(filename, sep='|', decimal=',')

if len(df.columns) < 2:
    df = pd.read_csv(filename, sep=';', decimal=',')

    if "float64" not in df.dtypes.values:
        df = pd.read_csv(filename, sep='|', decimal='.')

elif "float64" not in df.dtypes.values:
    df = pd.read_csv(filename, sep='|', decimal='.')

df.head(12)


# In[26]:


df.to_pickle('proj2_ex01.pkl')


# In[27]:


with open('proj2_scale.txt', 'r') as file:
    text = file.read()

words = text.split('\n')
print(words)


# In[37]:


df_copy = df.copy()
columns = []
for column in df_copy.columns:
    column_to_change = True
    for word in df_copy[column]:
        if word not in words:
            column_to_change = False
            break
    if column_to_change:
        columns.append(column)
        for index, word in enumerate(df_copy[column]):
            word = words.index(word)
            df_copy[column][index] = word + 1
print(columns)
df_copy


# In[29]:


df_copy.to_pickle('proj2_ex02.pkl')


# In[42]:


df_copy2 = df.copy()
print(columns)
print(words)
for column in columns:
    df_copy2[column] = df_copy2[column] = df_copy2[column].astype('category')
    df_copy2[column] = df_copy2[column].cat.set_categories(words)
df_copy2.dtypes


# In[41]:


df_copy2.to_pickle('proj2_ex03.pkl')


# In[35]:


new_df = df.copy()
new_df = new_df.select_dtypes(exclude=['float64'])
new_df.replace(to_replace=r'[^0-9\.\-\,]', value='', regex=True, inplace=True)
new_df.replace(to_replace=r'[\,]', value='.', regex=True, inplace=True)
new_df.replace(to_replace=r'^\.$', value='', regex=True, inplace=True)
new_df.replace(to_replace=r'^\-$', value='', regex=True, inplace=True)

new_df = new_df.apply(pd.to_numeric, errors='coerce')
new_df = new_df.dropna(axis=1, how='all')
new_df


# In[19]:


new_df.to_pickle('proj2_ex04.pkl')


# In[36]:


df_copy4 = df.copy()
df_copy4 = df_copy4.select_dtypes(include=['object'])
df_copy4 = df_copy4.loc[:, df_copy4.apply(pd.Series.nunique) <= 10]
df_copy4 = df_copy4.loc[:, df_copy4.apply(lambda x: x.str.contains('^[a-z]*$').all())]
df_copy4 = df_copy4.loc[:, ~df_copy4.isin(words).all()]

i = 1
for col in df_copy4.columns:
    df_copy4[col] = df_copy4[col].astype('category')
    result = pd.get_dummies(df_copy4[col])
    result.to_pickle(f'proj2_ex05_{i}.pkl')
    i += 1

print(result)


# In[ ]:




