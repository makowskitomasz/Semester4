#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import csv
import json
import numpy as np


# In[3]:


dataframes = []
for file in ('proj3_data1.json', 'proj3_data2.json', 'proj3_data3.json'):
    with open(file) as filename:
        data = pd.read_json(filename)
        dataframes.append(data)


# In[4]:


ex1_df = pd.concat(dataframes, ignore_index=True)


# In[5]:


ex1_df.to_json('ex01_all_data.json', orient='records')


# In[6]:


null_dict = {}
for index, row in ex1_df.iterrows():
    for column in ex1_df.columns:
        if pd.isna(row[column]):
            if column in null_dict.keys():
                null_dict[column] += 1
            else:
                null_dict[column] = 1
print(null_dict)


# In[7]:


with open('ex02_no_nulls.csv', 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for key, value in null_dict.items():
        writer.writerow([key, value])


# In[8]:


ex1_df['description'] = None
with open('proj3_params.json') as file:
    params = json.load(file)
concat_columns = params['concat_columns']
print(concat_columns)
for index, row in ex1_df.iterrows():
    tmp_string = str()
    for element in concat_columns:
        tmp_string += row[element]
        tmp_string += " "
    tmp_string = tmp_string[:-1]
    ex1_df.loc[index, 'description'] = tmp_string
ex1_df


# In[9]:


ex1_df.to_json('ex03_descriptions.json', orient='records')


# In[10]:


with open('proj3_more_data.json') as filename:
    ex4_df = pd.read_json(filename)

join_column = params['join_column']


# In[11]:


joined_df = pd.merge(ex1_df, ex4_df, on=join_column, how='left')
joined_df


# In[12]:


joined_df.to_json('ex04_joined.json', orient='records')


# In[41]:


int_columns = params['int_columns']
for index, row in joined_df.iterrows():
    tmp_string = row['description']
    tmp_string = tmp_string.lower()
    tmp_string = tmp_string.replace(' ', '_')
    print(tmp_string)
    row[joined_df.columns != 'description'].to_json(f'ex05_{tmp_string}.json')
    row.replace(np.nan, None, inplace=True)
    row[int_columns] = row[int_columns].astype('Int64')
    row[joined_df.columns != 'description'].to_json(f'ex05_int_{tmp_string}.json')


# In[23]:


aggregation = params['aggregations']
tmp_dict = dict()
for element in aggregation:
    agg_func_str = element[1]
    agg_func = getattr(pd.Series, agg_func_str)
    result = agg_func(joined_df[element[0]])
    tmp_string = f'{element[1]}_{element[0]}'
    tmp_dict[tmp_string] = result
    json_str = json.dumps(tmp_dict)
    with open('ex06_aggregations.json', 'w') as file:
        file.write(json_str)


# In[25]:


grouped_df = joined_df.groupby(params['grouping_column'])
grouped_df = grouped_df.filter(lambda x: len(x) > 1).groupby(params['grouping_column']).agg('mean', numeric_only=True)

grouped_df.to_csv('ex07_groups.csv', index=True, header=True)


# In[27]:


pivot_index = params['pivot_index']
pivot_columns = params['pivot_columns']
pivot_values = params['pivot_values']


# In[29]:


pivot_df = joined_df.pivot_table(index=pivot_index, columns=pivot_columns, values=pivot_values, aggfunc='max')
pivot_df.to_pickle('ex08_pivot.pkl')


# In[18]:


id_vars = params['id_vars']
value_vars = list(joined_df.columns)
value_vars = [element for element in value_vars if element not in id_vars]
df_long = joined_df.melt(id_vars=id_vars, value_vars=value_vars,var_name='variable', value_name='value')
df_long.to_csv('ex08_melt.csv', index=False, header=True)


# In[30]:


statistics_df = pd.read_csv('proj3_statistics.csv')
statistics_df


# In[36]:


columns = statistics_df.columns[1:].to_series().apply(lambda x : x.split('_')[0]).unique()
wide_long = pd.wide_to_long(statistics_df, columns, i=statistics_df.columns[0], j='suffixes', sep='_')
wide_long


# In[37]:


wide_long.to_pickle('ex08_stats.pkl')

