#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from sklearn.cluster import KMeans
data=pd.read_csv("pendigits_test.txt",delim_whitespace=True,skipinitialspace=True)
data = data.iloc[:,:-1]
X = data
y = data['88']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['88'] = le.fit_transform(X['88'])
y = le.transform(y)


# In[15]:


X.head()


# In[17]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
kmeans.cluster_centers_


# In[ ]:




