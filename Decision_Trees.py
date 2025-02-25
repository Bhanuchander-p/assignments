#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[4]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[5]:


iris.info()


# In[9]:


iris.isnull().sum()


# In[11]:


iris.isna().sum()


# In[12]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[ ]:




