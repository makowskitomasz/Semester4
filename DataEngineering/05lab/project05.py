#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd
import pickle


# In[ ]:


with open('proj5_params.json', 'r') as filename:
    parameters = json.load(filename)


# ## Exercise 1

# In[4]:


ex01_df = pd.read_csv('proj5_timeseries.csv')
ex01_df.columns = ex01_df.columns.str.lower()
ex01_df.columns = ex01_df.columns.str.replace('[^a-zA-Z]', '_', regex=True)
ex01_df.head()


# In[5]:


first_column = ex01_df.columns[0]
ex01_df[first_column] = pd.to_datetime(ex01_df[first_column])
ex01_df = ex01_df.set_index(first_column)
original_frequency = parameters['original_frequency']
ex01_df.head()


# In[6]:


ex01_df.index


# In[7]:


ex01_df = ex01_df.asfreq(original_frequency)


# In[8]:


ex01_df.index


# In[9]:


ex01_df.to_pickle('proj5_ex01.pkl')


# # Exercise 2

# In[10]:


ex02_df = ex01_df.copy()
print(ex02_df.head(5))
target_frequency = parameters['target_frequency']
print(target_frequency)
ex02_df = ex02_df.asfreq(target_frequency)
print(ex02_df.index)
ex02_df.to_pickle('proj5_ex02.pkl')


# ## Exercise 3

# In[11]:


print(parameters)


# In[12]:


downsample_periods = parameters['downsample_periods']
downsample_units = parameters['downsample_units']
print(downsample_periods, downsample_units)


# In[13]:


ex03_df = ex01_df.resample(f'{downsample_periods}{downsample_units}').sum(min_count=downsample_periods)
ex03_df


# In[14]:


ex03_df.to_pickle('proj5_ex03.pkl')


# ## Exercise 4

# In[15]:


upsample_periods = parameters['upsample_periods']
upsample_units = parameters['upsample_units']
interpolation = parameters['interpolation']
interpolation_order = parameters['interpolation_order']
print(upsample_periods, upsample_units, interpolation, interpolation_order)


# In[16]:


ex04_df = ex01_df.resample(f'{upsample_periods}{upsample_units}').interpolate(interpolation, order=interpolation_order)
print(ex04_df.index.freq / ex01_df.index.freq)
ex04_df = ex04_df * (ex04_df.index.freq / ex01_df.index.freq)


# In[17]:


ex04_df


# In[18]:


ex04_df.to_pickle('proj5_ex04.pkl')


# ## Exercise 5

# In[21]:


sensors_periods = parameters['sensors_periods']
sensors_units = parameters['sensors_units']
print(sensors_periods, sensors_units)


# In[22]:


ex05_df = pd.read_pickle('proj5_sensors.pkl')
ex05_df


# In[23]:


ex05_df = ex05_df.pivot(columns='device_id', values='value')
ex05_df


# In[24]:


new_index = pd.date_range(ex05_df.index.round(f'{sensors_periods}{sensors_units}').min(), ex05_df.index.round(f'{sensors_periods}{sensors_units}').max(), freq=f'{sensors_periods}{sensors_units}')
ex05_df = ex05_df.reindex(new_index.union(ex05_df.index)).interpolate()
ex05_df


# In[25]:


ex05_df = ex05_df.reindex(new_index)
ex05_df = ex05_df.dropna()
ex05_df


# In[50]:


ex05_df.to_pickle('proj5_ex05.pkl')

