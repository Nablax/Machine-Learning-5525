#!/usr/bin/env python
# coding: utf-8

# In[3]:


global k
import numpy as np
import pandas as pd
import cvxopt
from matplotlib import pyplot as plt
data_1=pd.read_csv("mfeat_train.csv",index_col=0).to_numpy()
data_2=pd.read_csv("mfeat_test.csv",index_col=0).to_numpy()
[m,n]=data_1.shape
#we have 10 classes, 1 - 10, and we want to make it 0-9, which is easier to operate
data_1[:,n-1]-=1
data_2[:,n-1]-=1
train_x=data_1[:,:n-1]
train_y=data_1[:,n-1].reshape(-1,1)
test_y=data_2[:,n-1].reshape(-1,1)
test_x=data_2[:,:n-1]
k=len(np.unique(train_y)) #k=10
print (np.unique(train_y))
print (np.unique(test_y))
#we need 10 datasets here to do 10 SVM, so we need to devide them based on their classes(labels)-->divide data function


# In[4]:


def rbf(X1,X2,sigma):
    return np.exp(-np.linalg.norm(X1-X2, axis=-1)**2/(2*sigma**2))


# In[5]:


def rbf_svm_train(X,y,c=1,sigma=1):#return Alpha
    [m, n]=X.shape
    y=y.reshape(-1,1)*1. #make it float
    Gram=np.zeros((m,m))
    count=0
    tmp = np.zeros((m, n))
    for i in range(m):
        tmp[:] = X[i]
        Gram[i]=rbf(tmp,X,sigma) #kernel
        count+=1
    cvxopt.solvers.options['show_progress'] = False
    P=cvxopt.matrix(np.outer(y,y)*Gram)
    q=cvxopt.matrix(-np.ones((m,1)))
    G=cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h=cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
    ans=cvxopt.solvers.qp(P,q,G,h)
    alpha=np.array(ans['x'])
    return alpha


# In[6]:


def predict(test_X,train_X,train_y,alpha,sigma=1):
    len1=test_X.shape[0]
    len2=train_X.shape[0]
    Gram=np.zeros((len1,len2))
    tmp = np.zeros((len2, 64))
    for i in range(len1):
        tmp[:] = test_X[i]
        Gram[i]=rbf(tmp,train_X,sigma)
    label= Gram @ (alpha*train_y)
    return label #predict based on 1 model


# In[7]:


def dividedata(data_1):
    temp=data_1.copy()
    xdict=dict()
    ydict=dict()
    for i in range(k):
        ydict[i]=np.where(temp[:,64]==i,1,0).reshape(-1,1) #1-10
        xdict[i]=temp[:,:64]
    return xdict,ydict #successful after test


# In[8]:


#print (xdict[0].shape)
#x_train=xdict[0]
#y_train=ydict[0]
#print (rbf_svm_train(x_train,y_train).shape)


# In[9]:


def getlabel(data_1,test_X): #we keep all trained weights in this step.
    xdict,ydict=dividedata(data_1)
    label=np.zeros((test_X.shape[0],1))
    final_weight=np.zeros((data_1.shape[0],1))
    for i in range(k):
        print (i)
        x_train=xdict[i]
        y_train=ydict[i].reshape(-1,1)
        alpha=rbf_svm_train(x_train,y_train) #predict on all models and combine
        #final_weight=np.concatenate((final_weight,alpha),axis=1) #combine all the alpha
        if(i==0):
            label=predict(test_X,x_train,y_train,alpha).reshape(-1,1)
            final_weight=alpha
        else:
            label=np.concatenate((label,predict(test_X,x_train,y_train,alpha).reshape(-1,1)),axis=1)
            final_weight=np.concatenate((final_weight,alpha.reshape(-1,1)),axis=1)
    y_pred=np.argmax(label, axis=1)
    return y_pred,final_weight #we do not need column #1 since it is used for intialization


# In[10]:


def err(label,test_y):
    return np.count_nonzero(label-test_y)/len(test_y)


# In[11]:


def confusion(label,test_y):
    conf=np.zeros((k,k))
    for i in range(len(label)):
        conf[int(test_y[i]),int(label[i])]+=1 #Actual and predicted class
    return conf


# In[12]:


label,weight=getlabel(data_1,test_x)
print (weight.shape)


# In[13]:


label=label.reshape(-1,1)
print (1-err(label,test_y))
print (confusion(label,test_y))
np.savetxt("problem7_weight",weight)


# In[ ]:




