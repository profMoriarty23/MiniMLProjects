#!/usr/bin/env python
# coding: utf-8

# In[24]:


import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ## Loading Data
# 

# In[14]:


data = load_breast_cancer()


# In[15]:


label_names = data["target_names"]
labels = data["target"]
feature_names = data["feature_names"]
features = data["data"]


# In[16]:


print(label_names)


# In[17]:


print(labels[0])
print(feature_names[0])
print(features[0])


# ## Splitting data

# In[18]:


train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42)


# # Building and evaluating the Model
# 

# In[20]:


# Initialize our classifier
gnb = GaussianNB()


# In[21]:


# Train our classifier
model = gnb.fit(train, train_labels)


# ## Make predictions
# 

# In[22]:


preds = gnb.predict(test)


# In[23]:


print(preds)


# ## Evaluating accuracy

# In[25]:


print(accuracy_score(test_labels, preds))


# In[ ]:




