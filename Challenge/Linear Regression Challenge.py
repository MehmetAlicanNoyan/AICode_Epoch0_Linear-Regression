#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


m = 5 # number of examples
n = 1 # number of features


# In[3]:


X = np.random.uniform(size=(m,n))
X


# In[4]:


X = np.concatenate((np.ones((m,1)), X), axis = 1)
X


# In[5]:


theta_true = np.random.uniform(size=(n+1,1))
theta_true


# In[6]:


y = np.dot(X,theta_true)
y


# In[7]:


plt.scatter(X[:,1],y)


# In[8]:


def plotter(X, y, theta):
    plt.scatter(X[:,1],y)
    linex = np.linspace(X[:,1].min(),X[:,1].max())
    liney = theta[0] + theta[1]*linex
    plt.plot(linex,liney)


# In[9]:


plotter(X, y, theta_true)


# In[10]:


theta = np.random.uniform(size=(n+1,1))
theta


# In[11]:


plotter(X, y, theta)


# In[12]:


def cost_function(X, y, theta):

    
    return cost


# In[13]:


cost_function(X, y, theta)


# In[14]:


cost_function(X, y, theta_true)


# In[15]:


# Simultaneous update is important
def GD_one_step(X, y, theta, lr):
    
    
    
    
    return theta_new


# In[16]:


print('theta: ', theta.T)
print('theta_true: ', theta_true.T)


# In[17]:


theta = GD_one_step(X, y, theta, lr=...)
theta


# In[18]:


def GD(X, y, lr, epoch):
    
    
    
    return theta


# In[19]:


theta = GD(X, y, lr=..., epoch=...)


# In[20]:


plotter(X, y, theta)


# In[21]:


def plotter_multiple(X, y, theta_multi):
    plt.scatter(X[:,1],y)
    
    for theta in theta_multi:
        linex = np.linspace(X[:,1].min(),X[:,1].max())
        liney = theta[0] + theta[1]*linex
        plt.plot(linex,liney)


# In[22]:


plotter_multiple(X, y, [theta, theta_true])


# In[23]:


# We need a way to monitor training
def GD_memory(X, y, lr, epoch):
    memory = []
    
    
        loss = cost_function(X, y, theta)
        memory.append(loss)
                
    return theta, memory


# In[24]:


theta, memory = GD_memory(X, y, lr=..., epoch=...)


# In[25]:


def loss_plotter(memory):
    plt.plot(memory)
    plt.ylabel('loss')
    plt.xlabel('number of epochs')


# In[26]:


loss_plotter(memory)


# In[27]:


# so far we have only fit ideal datapoints
# so let's get real
y_real = y * np.random.uniform(0.97,1.03,m).reshape(-1,1)


# In[28]:


plotter(X, y_real, theta_true)


# In[29]:


theta, memory = GD_memory(X, y_real, lr=..., epoch=...)


# In[30]:


loss_plotter(memory)


# In[31]:


plotter_multiple(X, y_real, [theta, theta_true])


# In[32]:


m = 5 # number of examples
n = 2 # number of features


# In[33]:


X = np.random.uniform(size=(m,n))
X


# In[34]:


X = np.concatenate((np.ones((m,1)), X), axis = 1)
X


# In[35]:


theta_true = np.random.uniform(size=(n+1,1))
theta_true


# In[36]:


y = np.dot(X, theta_true)


# In[37]:


theta, memory = GD_memory(X, y, lr=..., epoch=...)
loss_plotter(memory)


# In[38]:


print('theta: ', theta.T)
print('theta_true: ', theta_true.T)


# In[39]:


np.dot(X, theta)[0], y[0]


# In[ ]:


# Hints below


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def cost_function(X, y, theta):
    y_pred = np.dot(X, theta)
    cost = 
    return cost


# In[ ]:


# Simultaneous update is important
def GD_one_step(X, y, theta, lr):
    y_pred = np.dot(X, theta)
    theta_new = np.zeros(theta.shape)
    for i in range(n+1):
        theta_new[i] = theta[i] - #lr * derivative
    return theta_new


# In[ ]:


def GD(X, y, lr, epoch):
    theta = np.random.uniform(size=(n+1,1))
    for i in range(epoch):
        
    return theta

