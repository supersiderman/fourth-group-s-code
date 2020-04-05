#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np

import matplotlib.pylab as pyb
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets


# Loading data, dimension reduction mapping

# In[23]:


X,y = datasets.load_iris(True)
# 4 attributes, 4-dimensional space, 4-dimensional data 
# 150 represents the number of samples
X.shape


# In[24]:


# dimension reduction, slice
X=X[:,:2]
X.shape


# In[25]:


pyb.scatter(X[:,0],X[:,1],c=y)


# KNN algorithm training data

# In[27]:


Knn=KNeighborsClassifier(n_neighbors=5)

# training data using 150 samples
Knn.fit(X,y)


# In[31]:


# training data
X.shape
# Test data shape (?,2)


# Extract data ( 8000 samples )

# In[61]:


# get data 
# abscissa 4 ~ 8, ordinate 2 ~ 4.5 
# background point, take it out, meshgrid
x1=np.linspace(4,8,100)

y1=np.linspace(2,4.5,80)

X1,Y1=np.meshgrid(x1,y1)
##display(X1.shape,Y1.shape)
##pyb.scatter(X1,Y1)

#X1=X1.reshape(-1,1)

#Y1=Y1.reshape(-1,1)

#X_test=np.concatenate([X1,Y1],axis=1)
#X_test.shape
 
# one-dimensional
X_test=np.c_[X1.ravel(),Y1.ravel()]

X_test.shape




# Using algorithms to predict visualization

# In[35]:


a=np.random.randint(0,30,size=(3,4))
a


# In[36]:


a.ravel()


# In[39]:


get_ipython().run_cell_magic('time', '', 'y_=Knn.predict(X_test)')


# In[46]:


from matplotlib.colors import ListedColormap


# In[54]:


lc=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])

lc2=ListedColormap(['#FF0000','#00FF00','#0000FF'])


# In[56]:


get_ipython().run_cell_magic('time', '', 'pyb.scatter(X_test[:,0],X_test[:,1],c=y_,cmap=lc)\n\npyb.scatter(X[:,0],X[:,1],c=y,cmap=lc2)')


# In[62]:




pyb.contourf(X1,Y1,y_.reshape(80,100),cmap=lc)

pyb.scatter(X[:,0],X[:,1],c=y,cmap=lc2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets


# In[ ]:


##Categories continue to be broken down


# In[67]:


iris=datasets.load_iris()
iris
X=iris['data']

y=iris['target']

#One hundred and fifty samples and   four attributes：sepal length，sepal width，petal length，petal width

X.shape


# In[68]:


#Divide the data into two parts, train and test.
#Disrupt the order
index=np.arange(150)
index


# In[70]:


#Disrupt the order
np.random.shuffle(index)
index


# In[76]:



X_train,X_test=X[index[:100]],X[index[100:]]
y_train,y_test=y[index[:100]],y[index[-50:]]


# In[77]:


Knn=KNeighborsClassifier(n_neighbors=5)

Knn.fit(X_train,y_train)

y_=Knn.predict(X_test)


# In[78]:



print(y_)
print('---------')
print(y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


#get the data
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(x,y)


# In[82]:


X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=2003) 
clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,y_train)


# In[86]:


from sklearn.metrics import accuracy_score
correct=np.count_nonzero((clf.predict(X_test)==y_test)==True)
#accuracy
print ("Accuracy is :%.3f"%(correct/len(X_test)))
                         


# In[87]:


X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=2003) 
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)


# In[88]:


from sklearn.metrics import accuracy_score
correct=np.count_nonzero((clf.predict(X_test)==y_test)==True)
#accuracy
print ("Accuracy is :%.3f"%(correct/len(X_test)))


# In[89]:


X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=2003) 
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)


# In[90]:


from sklearn.metrics import accuracy_score
correct=np.count_nonzero((clf.predict(X_test)==y_test)==True)
#accuracy
print ("Accuracy is :%.3f"%(correct/len(X_test)))


# In[1]:


##When the k values are different, the performance indexes are different. 
##In order to select the optimal k value, the folding cross verification method is generally used.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




