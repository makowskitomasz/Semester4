#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle


# In[4]:


[test_set_raw, valid_set_row, train_set_raw], info = tfds.load('tf_flowers', split=['train[:10%]', 'train[10%:25%]', 'train[25%:]'], as_supervised=True, with_info=True)


# In[5]:


info


# In[6]:


class_names = info.features['label'].names
n_classes = info.features['label'].num_classes
dataset_size = info.splits['train'].num_examples
print(class_names, n_classes, dataset_size)


# In[7]:


plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9)
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title('Class: {}'.format(class_names[label]))
    plt.axis('off')

plt.show(block=False)


# In[8]:


def preprocess(_image, _label):
    resized_image = tf.image.resize(_image, [224, 224])
    return resized_image, _label


# In[9]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_row.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[10]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)

for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title('Class: {}'.format(class_names[y_batch[index]]))
        plt.axis('off')

plt.show()


# ## Exercise 2.2.2

# In[ ]:


rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)

train_set_scaled = train_set.map(lambda x, y: (rescale(x), y))
valid_set_scaled = valid_set.map(lambda x, y: (rescale(x), y))
test_set_scaled = test_set.map(lambda x, y: (rescale(x), y))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_set_scaled, epochs=10, validation_data=valid_set_scaled)


# In[12]:


acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]

result = (acc_train, acc_valid, acc_test)
print(result)


# In[ ]:


with open('simple_cnn_acc.pkl', 'wb') as filename:
    pickle.dump(result, filename)


# ## Exercise 2.3

# In[11]:


def preprocess(_image, _label):
    resized_image = tf.image.resize(_image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, _label


# In[12]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_row.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[13]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title('Class: {}'.format(class_names[y_batch[index]]))
        plt.axis('off')

plt.show()


# In[14]:


base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False)


# In[15]:


inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)


# In[ ]:


model = tf.keras.Model(inputs, outputs)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_set, epochs=5, validation_data=valid_set)

for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_set, epochs=5, validation_data=valid_set)


# In[ ]:


acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]

result = (acc_train, acc_valid, acc_test)
print(result)


# In[ ]:


with open('xception_acc.pkl', 'wb') as filename:
    pickle.dump(result, filename)

