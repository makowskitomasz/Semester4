#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import sqlite3


# In[4]:


con = sqlite3.connect('proj6_readings.sqlite')
cur = con.cursor()

result = cur.execute("select count(*) from readings;").fetchall()

df = pd.DataFrame(result)

df


# In[5]:


df = pd.read_sql('select count(*) from readings;', con)
df


# ## Exercise 1: Basic counting

# In[9]:


result1 = cur.execute('select count(distinct detector_id) from readings').fetchall()
ex01_df = pd.DataFrame(result1)
ex01_df


# In[10]:


ex01_df.to_pickle('proj6_ex01_detector_no.pkl')


# ## Exercise 2: Some stats for the detectors

# In[18]:


result2 = cur.execute('select detector_id, count(count), min(starttime), max(starttime) from readings group by(detector_id)'
                      ).fetchall()
ex02_df = pd.DataFrame(result2, columns=['detector_id', 'count(count)', 'min(starttime)', 'max(starttime']).reset_index(drop=True)
ex02_df


# In[20]:


ex02_df.to_pickle('proj6_ex02_detector_stat.pkl')


# ## Exercise 3: Moving Window

# In[21]:


query3 = '''
SELECT detector_id, count, LAG(count) OVER (PARTITION BY detector_id ORDER BY starttime) AS prev_count
FROM readings
WHERE detector_id = 146
LIMIT 500;
'''


# In[22]:


result3 = cur.execute(query3).fetchall()
ex03_df = pd.DataFrame(result3, columns=['detector_id', 'count', 'prev_count'])
ex03_df


# In[23]:


ex03_df.to_pickle('proj6_ex03_detector_146_lag.pkl')


# ## Exercise 4: Window

# In[39]:


query4 = '''
select detector_id, count, sum(count) over (partition by detector_id order by starttime rows between current row and 10 following) as window_sum
from readings
where detector_id = 146
limit 500;
'''


# In[40]:


result4 = cur.execute(query4).fetchall()
ex04_df = pd.DataFrame(result4, columns=['detector_id', 'count', 'window_sum'])
ex04_df


# In[41]:


ex04_df.to_pickle('proj6_ex04_detector_146_sum.pkl')

