#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neural_network import MLPClassifier


# In[4]:


iris = load_iris(as_frame=True).frame


# In[5]:


pd.concat([iris[iris.columns[:-1]], iris['target']], axis=1).plot.scatter(x='petal length (cm)', y='petal width (cm)', c='target',colormap='viridis', figsize=(10,4))


# In[6]:


X = iris.drop('target', axis=1)
y = iris['target']
X = X[['petal length (cm)', 'petal width (cm)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


results = []
accuracy_list_of_tuples = []
weight_list_of_tuples = []
for species in range(3):
    perceptron = Perceptron(max_iter=100, eta0=0.1)  # Example parameters
    perceptron.fit(X_train, (y_train == species).astype(int))

    train_accuracy = accuracy_score((y_train == species).astype(int), perceptron.predict(X_train))
    test_accuracy = accuracy_score((y_test == species).astype(int), perceptron.predict(X_test))
    class_biases = perceptron.intercept_
    class_weights = perceptron.coef_
    print(class_weights[0][0])
    results.append((train_accuracy, test_accuracy, class_biases, class_weights))
    accuracy_list_of_tuples.append((train_accuracy, test_accuracy))
    weight_list_of_tuples.append((class_biases[0], class_weights[0][0], class_weights[0][1]))

for species in range(3):
    print(f"Accuracy for species {species}:")
    print(f"- Training set: {results[species][0]:.2f}")
    print(f"- Test set: {results[species][1]:.2f}")
    print(f"Bias (w_0): {results[species][2]}")
    print(f"Weights (w_1, w_2): {results[species][3]}")
    print()

print(accuracy_list_of_tuples)
print(weight_list_of_tuples)


# In[8]:


with open('per_acc.pkl', 'wb') as file:
    pickle.dump(accuracy_list_of_tuples, file)

with open('per_wght.pkl', 'wb') as file:
    pickle.dump(weight_list_of_tuples, file)


# In[9]:


X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,1,1,0])


# In[10]:


perceptron = Perceptron(max_iter=100, eta0=0.1)
perceptron.fit(X, y)
accuracy = accuracy_score(y, perceptron.predict(X))
class_biases = perceptron.intercept_
class_weights = perceptron.coef_

print(f"- Accuracy: {accuracy:.2f}")
print(f"Bias (w_0): {class_biases}")
print(f"Weights (w_1, w_2): {class_weights}")
print()


# In[54]:


mlp = MLPClassifier(hidden_layer_sizes=(2,), solver='lbfgs', learning_rate='constant', activation='logistic', max_iter=1000)

average = 0

for _ in range(100):
    mlp.fit(X, y)
    average += mlp.score(X, y)
    
average /= 100

print(average)
print(mlp.intercepts_)
print(mlp.coefs_)


# In[55]:


with open('mlp_xor.pkl', 'wb') as file:
    pickle.dump(mlp, file)


# In[56]:


mlp.intercepts_ = [np.array([-1.5, 0.5]), np.array([-0.5])]
mlp.coefs_ = [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[-1.0], [1.0]])]
print(mlp.predict(X))
print(mlp.score(X, y))


# In[57]:


with open('mlp_xor_fixed.pkl', 'wb') as file:
    pickle.dump(mlp, file)

