#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import pickle


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True).frame


# In[3]:


X = data_breast_cancer.drop('target', axis=1)
y = data_breast_cancer['target']
X = X[['mean texture', 'mean symmetry']]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[5]:


# tree classifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
tree_train_predict = tree_clf.predict(X_train)
tree_test_predict = tree_clf.predict(X_test)

tree_train_accuracy = accuracy_score(y_train, tree_train_predict)
tree_test_accuracy = accuracy_score(y_test, tree_test_predict)

print(tree_train_accuracy, tree_test_accuracy)


# In[6]:


# logistic regression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_train_predict = log_clf.predict(X_train)
log_test_predict = log_clf.predict(X_test)

log_train_accuracy = accuracy_score(y_train, log_train_predict)
log_test_accuracy = accuracy_score(y_test, log_test_predict)

print(log_train_accuracy, log_test_accuracy)


# In[7]:


# k nearest neighbors
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_train_predict = knn_clf.predict(X_train)
knn_test_predict = knn_clf.predict(X_test)

knn_train_accuracy = accuracy_score(y_train, knn_train_predict)
knn_test_accuracy = accuracy_score(y_test, knn_test_predict)

print(knn_train_accuracy, knn_test_accuracy)


# In[8]:


# hard voting
hard_clf = VotingClassifier(estimators=[('lr', log_clf), ('tr', tree_clf), ('knn', knn_clf)], voting='hard')
hard_clf.fit(X_train, y_train)
hard_train_predict = hard_clf.predict(X_train)
hard_test_predict = hard_clf.predict(X_test)

hard_train_accuracy = accuracy_score(y_train, hard_train_predict)
hard_test_accuracy = accuracy_score(y_test, hard_test_predict)

print(hard_train_accuracy, hard_test_accuracy)


# In[9]:


# soft voting
soft_clf = VotingClassifier(estimators=[('lr', log_clf), ('tr', tree_clf), ('knn', knn_clf)], voting='soft')
soft_clf.fit(X_train, y_train)
soft_train_predict = soft_clf.predict(X_train)
soft_test_predict = soft_clf.predict(X_test)

soft_train_accuracy = accuracy_score(y_train, soft_train_predict)
soft_test_accuracy = accuracy_score(y_test, soft_test_predict)

print(soft_train_accuracy, soft_test_accuracy)


# In[10]:


print(log_train_accuracy, log_test_accuracy)
print(tree_train_accuracy, tree_test_accuracy)
print(knn_train_accuracy, knn_test_accuracy)
print(hard_train_accuracy, hard_test_accuracy)
print(soft_train_accuracy, soft_test_accuracy)


# In[11]:


accuracy_list = [(tree_train_accuracy, tree_test_accuracy),
                 (log_train_accuracy, log_test_accuracy),
                 (knn_train_accuracy, knn_test_accuracy),
                 (hard_train_accuracy, hard_test_accuracy),
                 (soft_train_accuracy, soft_test_accuracy)]
print(accuracy_list)


# In[12]:


with open('acc_vote.pkl', 'wb') as file:
    pickle.dump(accuracy_list, file)


# In[13]:


classificator_list = [tree_clf, log_clf, knn_clf, hard_clf, soft_clf]
print(classificator_list)


# In[14]:


with open('vote.pkl', 'wb') as file:
    pickle.dump(classificator_list, file)


# In[15]:


# bagging classifier
bag_clf = BaggingClassifier(n_estimators=30, bootstrap=True)
bag_clf.fit(X_train, y_train)
bag_train_predict = bag_clf.predict(X_train)
bag_test_predict = bag_clf.predict(X_test)

bag_train_accuracy = accuracy_score(y_train, bag_train_predict)
bag_test_accuracy = accuracy_score(y_test, bag_test_predict)

print(bag_train_accuracy, bag_test_accuracy)


# In[16]:


# bagging 50% instances
bag05_clf = BaggingClassifier(n_estimators=30, max_samples=0.5, bootstrap=True)
bag05_clf.fit(X_train, y_train)
bag05_train_predict = bag05_clf.predict(X_train)
bag05_test_predict = bag05_clf.predict(X_test)

bag05_train_accuracy = accuracy_score(y_train, bag05_train_predict)
bag05_test_accuracy = accuracy_score(y_test, bag05_test_predict)

print(bag05_train_accuracy, bag05_test_accuracy)


# In[17]:


# pasting classifier
pasting_clf = BaggingClassifier(n_estimators=30, bootstrap=False)
pasting_clf.fit(X_train, y_train)
pasting_train_predict = pasting_clf.predict(X_train)
pasting_test_predict = pasting_clf.predict(X_test)

pasting_train_accuracy = accuracy_score(y_train, pasting_train_predict)
pasting_test_accuracy = accuracy_score(y_test, pasting_test_predict)

print(pasting_train_accuracy, pasting_test_accuracy)


# In[18]:


# pasting 50% instances
pasting05_clf = BaggingClassifier(n_estimators=30, max_samples=0.5, bootstrap=False)
pasting05_clf.fit(X_train, y_train)
pasting05_train_predict = pasting05_clf.predict(X_train)
pasting05_test_predict = pasting05_clf.predict(X_test)

pasting05_train_accuracy = accuracy_score(y_train, pasting05_train_predict)
pasting05_test_accuracy = accuracy_score(y_test, pasting05_test_predict)

print(pasting05_train_accuracy, pasting05_test_accuracy)


# In[19]:


# random forrest classifier
rnd_clf = RandomForestClassifier(n_estimators=30)
rnd_clf.fit(X_train, y_train)
rnd_train_predict = rnd_clf.predict(X_train)
rnd_test_predict = rnd_clf.predict(X_test)

rnd_train_accuracy = accuracy_score(y_train, rnd_train_predict)
rnd_test_accuracy = accuracy_score(y_test, rnd_test_predict)

print(rnd_train_accuracy, rnd_test_accuracy)


# In[20]:


# adaBoost classifier
ada_clf = AdaBoostClassifier(n_estimators=30)
ada_clf.fit(X_train, y_train)
ada_train_predict = ada_clf.predict(X_train)
ada_test_predict = ada_clf.predict(X_test)

ada_train_accuracy = accuracy_score(y_train, ada_train_predict)
ada_test_accuracy = accuracy_score(y_test, ada_test_predict)

print(ada_train_accuracy, ada_test_accuracy)


# In[21]:


# gradient classifier
gradient_clf = GradientBoostingClassifier(n_estimators=30)
gradient_clf.fit(X_train, y_train)
gradient_train_predict = gradient_clf.predict(X_train)
gradient_test_predict = gradient_clf.predict(X_test)

gradient_train_accuracy = accuracy_score(y_train, gradient_train_predict)
gradient_test_accuracy = accuracy_score(y_test, gradient_test_predict)

print(gradient_train_accuracy, gradient_test_accuracy)


# In[22]:


bagging_accuracy_list = [
    (bag_train_accuracy, bag_test_accuracy),
    (bag05_train_accuracy, bag05_test_accuracy),
    (pasting_train_accuracy, pasting_test_accuracy),
    (pasting05_train_accuracy, pasting05_test_accuracy),
    (rnd_train_accuracy, rnd_test_accuracy),
    (ada_train_accuracy, ada_test_accuracy),
    (gradient_train_accuracy, gradient_test_accuracy)]


# In[23]:


print(bagging_accuracy_list)


# In[24]:


with open('acc_bag.pkl', 'wb') as file:
    pickle.dump(bagging_accuracy_list, file)


# In[25]:


bagging_classificator_list = [bag_clf, bag05_clf, pasting_clf, pasting05_clf, rnd_clf,
                               ada_clf, gradient_clf]


# In[26]:


print(bagging_classificator_list)


# In[27]:


with open('bag.pkl', 'wb') as file:
    pickle.dump(bagging_classificator_list, file)


# In[28]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True).frame
X = data_breast_cancer.drop('target', axis=1)
y = data_breast_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


pasting2_clf = BaggingClassifier(n_estimators=30, max_samples=0.5, max_features=2, bootstrap_features=False, bootstrap=True)

pasting2_clf.fit(X_train, y_train)
pasting2_train_predict = pasting2_clf.predict(X_train)
pasting2_test_predict = pasting2_clf.predict(X_test)

pasting2_train_accuracy = accuracy_score(y_train, pasting2_train_predict)
pasting2_test_accuracy = accuracy_score(y_test, pasting2_test_predict)

print(pasting2_train_accuracy, pasting2_test_accuracy)


# In[30]:


pasting2_accuracy_list = [pasting2_train_accuracy, pasting2_test_accuracy]


# In[31]:


with open('acc_fea.pkl', 'wb') as file:
    pickle.dump(pasting2_accuracy_list, file)


# In[32]:


pasting2_clf_list = [pasting2_clf]
print(pasting2_clf_list)


# In[33]:


with open('fea.pkl', 'wb') as file:
    pickle.dump(pasting2_clf_list, file)


# In[34]:


my_clf = BaggingClassifier()
my_clf.fit(X_train, y_train)
train_accuracy = []
test_accuracy = []
feature_names = []
for estimator in my_clf.estimators_:
    clf = BaggingClassifier(estimator, max_features=2)
    clf.fit(X_train, y_train)
    clf_train_predict = clf.predict(X_train)
    clf_test_predict = clf.predict(X_test)

    clf_train_accuracy = accuracy_score(y_train, clf_train_predict)
    clf_test_accuracy = accuracy_score(y_test, clf_test_predict)
    train_accuracy.append(clf_train_accuracy)
    test_accuracy.append(clf_test_accuracy)
    tmp_feature_names = []
    for name in clf.estimators_features_:
        tmp_tmp_feature_names = []
        for element in name:
            tmp_tmp_feature_names.append(X.columns[element])
        tmp_feature_names.append(tmp_tmp_feature_names)
    feature_names.append(tmp_feature_names)


# In[35]:


df = pd.DataFrame({'train_accuracy': train_accuracy,
                   'test_accuracy': test_accuracy,
                   'features': feature_names})


# In[36]:


df = df.sort_values(by=['test_accuracy', 'train_accuracy'], ascending=False)
df


# In[37]:


df.to_pickle('acc_fea_rank.pkl')

