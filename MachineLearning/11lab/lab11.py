#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
import scikeras
from scikeras.wrappers import KerasClassifier, KerasRegressor
from scipy.stats import reciprocal
import keras_tuner as kt
import pickle


# In[2]:


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[3]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[4]:


param_distribs = {
    "model__n_hidden": [1, 2, 3, 4],
    "model__n_neurons": np.arange(1, 100),
    "model__learning_rate": [3e-4, 3e-3, 3e-2],
    "model__optimizer": ['sgd', 'nesterov', 'momentum', 'adam']
}


# In[5]:


def build_model(n_hidden=1, n_neurons=100, optimizer='sgd', learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_neurons, activation='relu', input_shape=(8,)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
    model.add(keras.layers.Dense(1))
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'nesterov':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    elif optimizer == 'momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# In[6]:


es = keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[7]:


rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)


# In[8]:


best_parameters = rnd_search_cv.best_params_
param_distribs["model__optimizer"] = best_parameters['model__optimizer']
param_distribs['model__n_neurons'] = best_parameters['model__n_neurons']
param_distribs['model__n_hidden'] = best_parameters['model__n_hidden']
param_distribs["model__learning_rate"] = best_parameters["model__learning_rate"]
print(param_distribs)


# In[9]:


with open('rnd_search_params.pkl', 'wb') as filename:
    pickle.dump(param_distribs, filename)


# In[10]:


with open('rnd_search_scikeras.pkl', 'wb') as filename:
    pickle.dump(rnd_search_cv, filename)


# In[11]:


def build_model_kt(hp):
    n_hidden = hp.Int('n_hidden', min_value=1, max_value=3, default=2)
    n_neurons = hp.Int('n_neurons', min_value=1, max_value=100, default=30)
    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-2, default=3e-3)
    optimizer = hp.Choice('optimizer', values=['sgd', 'nesterov', 'momentum', 'adam'], default='sgd')

    if optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    elif optimizer == "momentum":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model = keras.Sequential()
    model.add(keras.layers.Dense(n_neurons, activation="relu", input_shape=(8,)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=optimizer, loss="mse", metrics=['mse'])
    return model


# In[12]:


random_search_tuner = kt.RandomSearch(build_model_kt, objective='val_mse', max_trials=10, overwrite=True, directory='my_california_housing', project_name='my_rnd_search', seed=42)
root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = keras.callbacks.TensorBoard(root_logdir)


# In[14]:


random_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tb, es])


# In[16]:


best_parameters = random_search_tuner.get_best_hyperparameters(num_trials=1)[0].values
tmp_distribs = dict()
tmp_distribs['n_hidden'] = best_parameters['n_hidden']
tmp_distribs['n_neurons'] = best_parameters['n_neurons']
tmp_distribs['learning_rate'] = best_parameters['learning_rate']
tmp_distribs['optimizer'] = best_parameters['optimizer']
print(tmp_distribs)


# In[17]:


with open('kt_search_params.pkl', 'wb') as filename:
    pickle.dump(tmp_distribs, filename)


# In[18]:


best_model = random_search_tuner.get_best_models(num_models=1)[0]
best_model.save('kt_best_model.h5')

