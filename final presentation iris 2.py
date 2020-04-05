#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


#get the data of iris from sklearn
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(x,y)


# In[6]:


### Iris Setosa（山鸢尾） 
    #Iris Versicolour（杂色鸢尾）
    #Iris Virginica（维吉尼亚鸢尾）


# In[8]:



import pandas
#Import the dataset of iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) #Read the data from csv
print(dataset.describe())


# In[9]:


import pandas
#import the dataset of iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csdv(url, names=names) #Read csv data
print(dataset.describe())
# histograms
dataset.hist()


# In[10]:


import pandas
#import the dateset of iris  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) #Read csv data
print(dataset.describe())
dataset.plot(x='sepal-length', y='sepal-width', kind='scatter')


# In[11]:


import pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) #Read csv data
print(dataset.describe())
dataset.plot(kind='kde')


# In[14]:


import pandas 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) #Read csv data
print(dataset.describe())
dataset.plot(kind='kde')
dataset.plot(kind='box', subplots=True, layout=(2,2), 
             sharex=False, sharey=False)


# In[ ]:




