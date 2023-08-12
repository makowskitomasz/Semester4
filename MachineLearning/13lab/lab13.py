#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
import pickle


# In[2]:


tf.keras.utils.get_file("bike_sharing_dataset.zip","https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",cache_dir=".",extract=True)


# In[3]:


# def date_parser(x):
#     return pd.datetime.strptime(x, '%Y-%m-%d %H')
#
# df = pd.read_csv('datasets/hour.csv', parse_dates={'datetime': ['dteday', 'hr']}, date_parser=date_parser, index_col='datetime')

df = pd.read_csv('datasets/hour.csv', parse_dates={'datetime': ['dteday', 'hr']}, date_format='%Y-%m-%d %H', index_col='datetime')


# In[4]:


print(df.index.min(), df.index.max())


# In[5]:


(365 + 366) * 24 - len(df)


# In[6]:


columns_to_drop = ['instant', 'season', 'yr', 'mnth']
df.drop(columns_to_drop, axis=1, inplace=True)


# In[7]:


df = df.resample('H').asfreq()

columns_to_fill_zero = ['casual', 'registered', 'cnt']
df[columns_to_fill_zero] = df[columns_to_fill_zero].fillna(0)

columns_to_interpolate = ['temp', 'atemp', 'hum', 'windspeed']
df[columns_to_interpolate] = df[columns_to_interpolate].interpolate()

columns_to_fill_previous = ['holiday', 'weekday', 'workingday', 'weathersit']
df[columns_to_fill_previous] = df[columns_to_fill_previous].fillna(method='ffill')


# In[8]:


print((365 + 366) * 24 - len(df))
df.notna().sum()


# In[9]:


df[['casual', 'registered', 'cnt', 'weathersit']].describe()


# In[10]:


df.casual/=1e3
df.registered/=1e3
df.cnt/=1e3
df.weathersit/=4


# In[11]:


df_2weeks=df[:24*7*2]
df_2weeks[['casual','registered','cnt','temp']].plot(figsize=(10,3))


# In[12]:


df_daily=df.resample('W').mean()
df_daily[['casual','registered','cnt','temp']].plot(figsize=(10,3))


# ## Exercise 2.3

# In[13]:


previous_day_prediction = df['cnt'].shift(24)
mae_daily = np.mean(np.abs(df['cnt'] - previous_day_prediction))

previous_week_prediction = df['cnt'].shift(24 * 7)
mae_weekly = np.mean(np.abs(df['cnt'] - previous_week_prediction))

mae_daily *= 1e3
mae_weekly *= 1e3

mae_values = (mae_daily, mae_weekly)
print(mae_values)


# In[14]:


with open("mae_baseline.pkl", "wb") as filename:
    pickle.dump(mae_values, filename)


# ## Exercise 2.4

# In[15]:


cnt_train=df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid=df['cnt']['2012-07-01 00:00':]


# In[16]:


seq_len=1*24

train_ds=tf.keras.utils.timeseries_dataset_from_array(cnt_train.to_numpy(),targets=cnt_train[seq_len:],sequence_length=seq_len,batch_size=32,shuffle=True,seed=42)

valid_ds=tf.keras.utils.timeseries_dataset_from_array(cnt_valid.to_numpy(),targets=cnt_valid[seq_len:],sequence_length=seq_len,batch_size=32)


# In[17]:


model=tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[seq_len])])


# In[18]:


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss=tf.keras.losses.Huber(), metrics=['mae'])

history = model.fit(train_ds, epochs=20, validation_data=valid_ds)


# In[19]:


model.save('model_linear.h5')
valid_mae = history.history['val_mae'][-1]
valid_mae_tuple = (valid_mae,)
print(valid_mae_tuple)


# In[20]:


with open('mae_linear.pkl', 'wb') as filename:
    pickle.dump(valid_mae_tuple, filename)


# ## Exercise 2.5

# In[21]:


model = tf.keras.Sequential([tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
              loss=tf.keras.losses.Huber(), metrics=['mae'])

history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

model.save('model_rnn1.h5')

valid_mae_rnn1 = history.history['val_mae'][-1]
valid_mae_rnn1_tuple = (valid_mae_rnn1,)
print(valid_mae_rnn1_tuple)


# In[22]:


print(valid_mae_rnn1)
with open('mae_rnn1.pkl', 'wb') as filename:
    pickle.dump(valid_mae_rnn1_tuple, filename)


# In[23]:


model = tf.keras.Sequential([tf.keras.layers.SimpleRNN(32, input_shape=[None, 1], return_sequences=True),
                             tf.keras.layers.Dense(1)])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
              loss=tf.keras.losses.Huber(), metrics=['mae'])

history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

model.save('model_rnn32.h5')

valid_mae_rnn32 = history.history['val_mae'][-1]
valid_mae_rnn32_tuple = (valid_mae_rnn32,)
print(valid_mae_rnn32_tuple)


# In[24]:


with open('mae_rnn32.pkl', 'wb') as filename:
    pickle.dump(valid_mae_rnn32_tuple, filename)


# ## Exercise 2.6

# In[25]:


model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1], return_sequences=True),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32, return_sequences=False),
    tf.keras.layers.Dense(1)])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
              loss=tf.keras.losses.Huber(), metrics=['mae'])

history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

model.save('model_rnn_deep.h5')

valid_mae_rnn_deep = history.history['val_mae'][-1]
valid_mae_rnn_deep_tuple = (valid_mae_rnn_deep,)
print(valid_mae_rnn_deep_tuple)


# In[26]:


with open('mae_rnn_deep.pkl', 'wb') as filename:
    pickle.dump(valid_mae_rnn_deep_tuple, filename)


# ## Exercise 2.7

# In[27]:


multi_train = df.loc['2011-01-01 00:00':'2012-06-30 23:00', ['cnt', 'atemp', 'workingday', 'weathersit']]
multi_valid = df.loc['2012-07-01 00:00':, ['cnt', 'atemp', 'workingday', 'weathersit']]
seq_len = 1 * 24

train_ds = tf.keras.utils.timeseries_dataset_from_array(multi_train.to_numpy(), targets=multi_train[seq_len:], sequence_length=seq_len, batch_size=32, shuffle=True, seed=42)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(multi_valid.to_numpy(), targets=multi_valid[seq_len:], sequence_length=seq_len, batch_size=32)


# In[28]:


number_of_features = 4
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, number_of_features], return_sequences=False),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
              loss=tf.keras.losses.Huber(), metrics=['mae'])

history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

model.save('model_rnn_mv.h5')

valid_mae_rnn_mv = history.history['val_mae'][-1]
valid_mae_rnn_mv_tuple = (valid_mae_rnn_mv,)
print(valid_mae_rnn_mv_tuple)


# In[29]:


with open('mae_rnn_mv.pkl', 'wb') as filename:
    pickle.dump(valid_mae_rnn_mv_tuple, filename)

