#!/usr/bin/env python
# coding: utf-8

# 2.1 Pobieranie danych
# Pobierz zestaw danych California Housing i dokonaj jego podziału oraz normalizacji:

# In[10]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# Celem ćwiczenia jest przejrzenie przestrzeni parametrów w następujących zakresach:  
# 1. krok uczenia: [3 ⋅ 10−4, 3 ⋅ 10−2],  
# 2. liczba warstw ukrytych: od 0 do 3,  
# 3. liczba neuronów na warstwę: od 1 do 100,  
# 4. algorytm optymalizacji: adam, sgd lub nesterov.  
# 
# W tym ćwiczeniu wykorzystamy narzędzie RandomizedSearchCV pakietu scikit-learn.  
# Aby móc go użyć, należy nasz model obudować wrapperem scikeras.  
# Przygotuj słownik zawierający przeszukiwane wartości parametrów:  
# ```python
# param_distribs = {  
# "model__n_hidden": ...,  
# "model__n_neurons": ...,  
# "model__learning_rate": ...,  
# "model__optimizer": ...  
# }  
# ```

# In[11]:


import numpy as np
from scipy.stats import reciprocal

param_distribs = {
    'model__n_hidden': [1, 2, 3], 
    'model__n_neurons': np.arange(1, 100),
    'model__optimizer': ['sgd', 'nesterov', 'momentum', 'adam'],
    'model__learning_rate' : reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
    }


# Przygotuj funkcję:
# ```python
# def build_model(n_hidden, n_neurons, optimizer, learning_rate):  
#     model = tf.keras.models.Sequential()  
#     ...  
#     model.compile(...)  
#     return model  
# ```
# budującą model według parametrów podanych jako argumenty:  
# • n_hidden – liczba warstw ukrytych,  
# • n_neurons – liczba neuronów na każdej z warstw ukrytych,  
# • optimizer – gradientowy algorytm optymalizacji, funkcja powinna rozumieć wartości: sgd,  
# nesterov, momentum oraz adam,  
# • learning_rate – krok uczenia.  
# 

# In[12]:


from tensorflow import keras

def build_model(n_hidden=1, n_neurons=30, optimizer="sgd", learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_neurons, activation="relu", input_shape=(8,)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    
    if optimizer == "sgd":
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss="mse")
    elif optimizer == "adam":
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    elif optimizer == "nesterov":
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True), loss="mse")
    elif optimizer == "momentum":
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9), loss="mse")
    
    return model


# Przygotuj callback early stopping i obuduj przygotowaną wcześniej funkcję build_model obiektem  
# KerasRegressor

# In[13]:


from scikeras.wrappers import KerasRegressor

es = keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])


# Przygotuj obiekt RandomizedSearchCV, tak aby wykonał 10 iteracji przy 3-krotnej walidacji  
# krzyżowej, a następnie przeprowadź uczenie:

# In[14]:


from sklearn.model_selection import RandomizedSearchCV

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))


# Zapisz najlepsze znalezione parametry w postaci słownika do pliku rnd_search_params.pkl w  
# postaci słownika o następującej strukturze:  
# ```python
# {'model__optimizer': 'adam',
# 'model__n_neurons': 42,
# 'model__n_hidden': 3,
# 'model__learning_rate': 0.004003820130936959}
# ```

# In[15]:


import pickle
best_params = rnd_search_cv.best_params_
best_params_dict = {
    'model__optimizer': best_params['model__optimizer'],
    'model__n_neurons': best_params['model__n_neurons'],
    'model__n_hidden': best_params['model__n_hidden'],
    'model__learning_rate': best_params['model__learning_rate']
}

with open('rnd_search_params.pkl', 'wb') as f:
    pickle.dump(best_params_dict, f)


# Zapisz obiekt RandomizedSearchCV do pliku rnd_search_scikeras.pkl.

# In[16]:


with open('rnd_search_scikeras.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv, f)


# 2.3 Przeszukiwanie przestrzeni hiperparametrów przy pomocy Keras Tuner  
# Przeprowadź podobny eksperyment przy pomocy KerasTuner. Przyjmij identyczne jak w poprzed-  
# nim ćwiczeniu zakresy hiperparametrów.  
# Przygotuj funkcję build_model_kt, przyjmującą obiekt HyperParameters jako wejście. Powinna  
# ona w pierwszej części definiować hiperparametry, a w drugiej – przeprowadzić budowę modelu:  
# ```python
# import keras_tuner as kt
# def build_model_kt(hp):
#     n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
#     # (...)
#     model = tf.keras.models.Sequential()
#     # (...)
#     # model.compile(...)
#     return model

# In[74]:


def build_model_kt(hp):
    n_hidden = hp.Int('n_hidden', min_value=0, max_value=3, default=2)
    n_neurons = hp.Int('n_neurons', min_value=1, max_value=100, default=30)
    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-2)
    optimizer = hp.Choice('optimizer', values=['sgd', 'nesterov', 'momentum', 'adam'])
    
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


# Przygotuj wybrany tuner spośród dostępnych w Keras Tuner, np.:

# In[75]:


import keras_tuner as kt
random_search_tuner = kt.RandomSearch(build_model_kt, objective="val_mse", max_trials=10, overwrite=True,directory="my_california_housing", project_name="my_rnd_search", seed=42)


# Przygotuj również callback TensorBoard do zbierania danych w podkatalogu tensorboard w kat-  
# alogu projektu:

# In[76]:


import os

root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
print(root_logdir)
tb = keras.callbacks.TensorBoard(root_logdir)


# Uruchom przeszukiwanie dla maksymalnie 100 epok na próbę. Pamiętaj o podaniu danych wali-  
# dacyjnych (X_valid, y_valid) oraz utworzonego przed chwilą callbacku TensorBoard oraz stwor-  
# zonego wcześniej callbacku early stopping.  

# In[77]:


random_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[tb, es])


# Uruchom TensorBoard i przeanalizuj proces strojenia hiperparametrów w zakładce HPARAMS:  
# %load_ext tensorboard  
# %tensorboard --logdir {root_logdir}  

# In[78]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir=./my_california_housing --port=6007')


# Zapisz do pliku kt_search_params.pkl parametry najlepszego znalezionego modelu w postaci  
# słownika, np.:
# ```python
# {'n_hidden': 3,
# 'n_neurons': 45,
# 'learning_rate': 0.0008960175671873151,
# 'optimizer': 'adam'}

# In[82]:


best_params = random_search_tuner.get_best_hyperparameters(num_trials=1)[0].values
print(best_params)

best_params_dict = {'n_hidden': best_params['n_hidden'],
'n_neurons': best_params['n_neurons'],
'learning_rate': best_params['learning_rate'],
'optimizer': best_params['optimizer']}

with open('kt_search_params.pkl', 'wb') as f:
    pickle.dump(best_params_dict, f)
    
# Zapisz najlepszy uzyskany model do pliku kt_best_model.h5.
best_model = random_search_tuner.get_best_models(num_models=1)[0]
best_model.save('kt_best_model.h5')


