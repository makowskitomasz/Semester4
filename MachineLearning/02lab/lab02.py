#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1)


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import time
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# In[4]:


print(mnist.keys())
print(mnist["target"][0])


# In[5]:


print((np.array(mnist.data.loc[0]).reshape(28, 28) > 0).astype(int))


# In[6]:


X, y = mnist["data"], mnist["target"].astype(np.uint8)
y = y.sort_values(ascending=True)
X = X.reindex(y.index)


# In[7]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[8]:


print(np.unique(y_train))
print(np.unique(y_test))


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[10]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
print(y_train_0.head())
print(np.unique(y_train_0))
print(len(y_train_0))


# In[11]:


start = time.time()
sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_0)
print(time.time() - start)


# In[12]:


start = time.time()
print(sgd_classifier.predict([mnist.data.loc[0], mnist.data.loc[1]]))
print(time.time()-start)


# In[13]:


y_train_pred = sgd_classifier.predict(X_train)
y_test_pred = sgd_classifier.predict(X_test)

acc_train = sum(y_train_pred == y_train_0)/len(y_train_0)
acc_test = sum(y_test_pred == y_test_0)/len(y_test_0)

print(acc_train, acc_test)
acc_list = [acc_train, acc_test]
print(acc_list)

with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(acc_list, f)


# In[14]:


start = time.time()
score = cross_val_score(sgd_classifier, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(time.time() - start)
print(score)


# In[17]:


sgd_m_clf = SGDClassifier(random_state=42,n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)
print(sgd_m_clf.predict([mnist.data.loc[0],
mnist.data.loc[1]]))


# In[24]:


print(cross_val_score(sgd_m_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1))
y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)
confusion_matrix = confusion_matrix(y_train, y_train_pred)
print(confusion_matrix)


# In[26]:


with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(score, f)


# In[27]:


with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(confusion_matrix, f)

