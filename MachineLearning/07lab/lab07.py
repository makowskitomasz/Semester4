#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_openml
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN


# In[3]:


mnist= fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target=mnist.target.astype(np.uint8)
X=mnist["data"]
y=mnist["target"]


# In[4]:


silhouette_array = []
labels10 = list()
k_array = [8, 9, 10, 11, 12]
for k in k_array:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_predict = kmeans.fit_predict(X)
    k_silhouette = silhouette_score(X, kmeans.labels_)
    print(f'{k}-centroids: {k_silhouette}')
    silhouette_array.append(k_silhouette)
    if k == 10:
        labels10 = kmeans.labels_
print(silhouette_array)


# In[5]:


with open('kmeans_sil.pkl', 'wb') as filename:
    pickle.dump(silhouette_array, filename)


# In[6]:


conf_mat = confusion_matrix(y, labels10)
print(conf_mat)


# In[8]:


argmax_array = []
for row in conf_mat:
    argmax_array.append(np.argmax(row))
print(argmax_array)
argmax_array.sort()
argmax_array = set(argmax_array)
argmax_array = list(argmax_array)
print(argmax_array)


# In[9]:


with open('kmeans_argmax.pkl', 'wb') as filename:
    pickle.dump(argmax_array, filename)


# In[10]:


distances = []
for i in range(300):
    for j in range(X.shape[0]):
        if i != j:
            distance = np.linalg.norm(X[i] - X[j])
            if distance == 0:
                continue
            else:
                distances.append(distance)
distances.sort()
distances = distances[:10]
print(distances)


# In[11]:


with open('dist.pkl', 'wb') as filename:
    pickle.dump(distances, filename)


# In[ ]:


mean_distance = (distances[0] + distances[1] + distances[2]) / 3.0
end = 1.1 * mean_distance
step = 0.04 * mean_distance
tmp_value = mean_distance
dbscan_array = []
while tmp_value <= end:
    dbscan = DBSCAN(eps=tmp_value)
    dbscan.fit(X)
    print(dbscan.labels_[:15])
    labels = dbscan.labels_
    unique_labels = len(np.unique(labels))
    dbscan_array.append(unique_labels)
    tmp_value += step


# In[1]:


print(dbscan_array)
with open('dbscan_len.pkl', 'wb') as filename:
    pickle.dump(dbscan_array, filename)

