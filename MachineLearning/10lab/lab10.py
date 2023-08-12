#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape==(60000,28,28)
assert X_test.shape==(10000,28,28)
assert y_train.shape==(60000,)
assert y_test.shape==(10000,)


# In[4]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[5]:


plt.imshow(X_train[142], cmap='binary')
plt.axis('off')
plt.show()


# In[6]:


class_names=["koszulka","spodnie","pulower","sukienka","kurtka","sandał","koszula","but", "torba","kozak"]
class_names[y_train[142]]


# In[7]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


# In[8]:


model.summary()


# In[9]:


tf.keras.utils.plot_model(model,"fashion_mnist.png", show_shapes=True)


# In[10]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[11]:


print(os.getcwd())


# In[12]:


def new_folder(folder_name):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(folder_name, current_time)
    print(log_dir)
    return log_dir


# In[13]:


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=new_folder('image_logs'), histogram_freq=1)


# In[14]:


epochs = 20
history = model.fit(X_train, y_train, epochs=epochs,validation_split=0.1, callbacks=[tensorboard_callback])


# In[15]:


for _ in range(5):
    image_index=np.random.randint(len(X_test))
    image=np.array([X_test[image_index]])
    confidences=model.predict(image)
    confidence=np.max(confidences[0])
    prediction=np.argmax(confidences[0])
    print("Prediction:", class_names[prediction])
    print("Confidence:", confidence)
    print("Truth:", class_names[y_test[image_index]])
    plt.imshow(image[0], cmap="binary")
    plt.axis('off')
    plt.show()


# In[16]:


model.save('fashion_clf.h5')


# In[17]:


housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
validation_split = 0.1
validation_samples = int(len(X_train) * validation_split)
X_val = X_train[:validation_samples]
y_val = y_train[:validation_samples]
X_train = X_train[validation_samples:]
y_train = y_train[validation_samples:]
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)


# In[18]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)


# In[19]:


regression_model = keras.models.Sequential()
regression_model.add(keras.layers.Dense(30, activation='relu', input_shape=(X_train.shape[1],)))
regression_model.add(keras.layers.Dense(1))
regression_model.compile(loss='mean_squared_error', optimizer='sgd')


# In[20]:


early_stopping = keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# In[21]:


tensorboard_callback_regression = keras.callbacks.TensorBoard(log_dir=new_folder('housing_logs/regression_model'), histogram_freq=1)


# In[22]:


epochs = 100
regression_model.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, tensorboard_callback_regression]
)


# In[23]:


regression_model.save('reg_housing_1.h5')


# In[24]:


regression_model2 = keras.models.Sequential()
regression_model2.add(keras.layers.Dense(64, activation='relu'))
regression_model2.add(keras.layers.Dense(64, activation='relu'))
regression_model2.add(keras.layers.Dense(1))
regression_model2.compile(optimizer='sgd', loss='mean_squared_error')


# In[25]:


tensorboard_callback_regression = keras.callbacks.TensorBoard(log_dir=new_folder('housing_logs/regression_model2'), histogram_freq=1)


# In[26]:


epochs = 1000
history2 = regression_model2.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, tensorboard_callback_regression]
)


# In[27]:


regression_model2.save('reg_housing_2.h5')


# In[28]:


regression_model3 = keras.models.Sequential()
regression_model3.add(keras.layers.Dense(256, activation='relu'))
regression_model3.add(keras.layers.Dense(128, activation='relu'))
regression_model3.add(keras.layers.Dense(64, activation='relu'))
regression_model3.add(keras.layers.Dense(32, activation='relu'))
regression_model3.add(keras.layers.Dense(16, activation='relu'))
regression_model3.add(keras.layers.Dense(1))
regression_model3.compile(optimizer='sgd', loss='mean_squared_error')


# In[29]:


tensorboard_callback_regression = keras.callbacks.TensorBoard(log_dir=new_folder('housing_logs/regression_model3'), histogram_freq=1)


# In[30]:


epochs = 1000
history3 = regression_model3.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, tensorboard_callback_regression]
)


# In[31]:


regression_model3.save('reg_housing_3.h5')

