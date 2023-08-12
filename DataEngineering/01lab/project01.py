#!/usr/bin/env python
# coding: utf-8

# In[170]:


import csv
import pandas as pd
import json
import re
import openpyxl
import tabulate


# In[171]:


df = pd.read_csv('lab1_ex01.csv')


# In[172]:


def df_to_json(dataframe, filename):
    list_of_dicts = []
    for column in df.columns:
        tmp_dict = {}
        tmp_dict["name"] = df[column].name
        missing = dataframe[column].isnull().sum() / len(df)
        tmp_dict["missing"] = missing
        type_of_data = dataframe[column].dtype
        if type_of_data == "float64":
            tmp_dict["type"] = "float"
        elif type_of_data == "int64":
            tmp_dict["type"] = "int"
        else:
            tmp_dict["type"] = "other"
        list_of_dicts.append(tmp_dict)
                
    with open(filename, 'w') as file:
        json.dump(list_of_dicts, file)
        
    


# In[173]:


df_to_json(df, "ex01_fields.json")


# In[174]:


def statistics_for_all_columns(dataframe, filename):
    df_statistics = df.describe(include='all')
    df_statistics = df_statistics.dropna(axis=1, how='all')
    df_dictionary = df_statistics.to_dict()
    for element in df_dictionary:
        new_dict = {}
        for key, value in df_dictionary[element].items():
            if not pd.isnull(value):
                new_dict[key] = value
        df_dictionary[element] = new_dict
    
    with open(filename, 'w') as file:
        json.dump(df_dictionary, file)


# In[175]:


statistics_for_all_columns(df, "ex02_stats.json")


# In[176]:


for column in df.columns:
    pattern = r"[^A-Za-z0-9_ ]"
    modified_string = re.sub(pattern, '', column).lower().replace(" ", "_")
    df.rename(columns = {column : modified_string}, inplace = True)

df.to_csv("ex03_columns.csv", index = False)


# In[177]:


df.to_json("ex04_json.json", orient="records")
df.to_excel("ex04_excel.xlsx", index = False)
df.to_pickle("ex04_pickle.pkl")


# In[178]:


ex05_dataframe = pd.read_pickle("lab1_ex05.pkl")

specified_columns = ex05_dataframe.iloc[:,[1, 2]]
rows_starting_with_v = specified_columns.index.str.startswith("v")
specified_columns = specified_columns[rows_starting_with_v]

with open("ex05_table.md", 'w') as file:
    file.write(tabulate.tabulate(specified_columns.fillna(""), headers='keys', tablefmt='pipe', showindex=True, missingval=""))


# In[179]:


ex06_dataframe = pd.read_json("lab1_ex06.json")
ex06_dataframe = pd.json_normalize(ex06_dataframe.to_dict(orient="records"))

ex06_dataframe.to_pickle("ex06_pickle.pkl")

