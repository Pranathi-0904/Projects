#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt
def randomNumbers(start, stop, numberOfPoints):
    d1 = np.random.uniform(start, stop, numberOfPoints)
    d2 = np.random.uniform(start, stop, numberOfPoints)
    return d1, d2

def function(d1,d2):
    d3 = d1 + 2*d2 -2
    d4 = np.sign(d3)
    d4 = d4.astype(int)
    return d4
def predictfunction(w,x1,x2):
    y1 = w[0] + w[1]*x1 + w[2]*x2
    y1 = np.sign(y1)
    y1 = y1.astype(int)
    return y1

def pla(d1,d2,res):
    w = rand(2) + 0.5

    predicted = predictfunction(w, d1, d2)
    MissPredictions = np.sum(predicted != res)
    count = 0
    while(MissPredictions > 0):
        count = count + 1
        isPredictionWrong = (predicted != res)
        occurences = np.where(isPredictionWrong==True)
        firstIndex = occurences[0][0]
        x = np.array([1, d1[firstIndex], d2[firstIndex]])
        y = res[firstIndex]

        for i in range(w.size):
            temp = w[i] + y*x[i]
            w[i] = temp
        
        predicted = predictfunction(w, d1, d2)
        numOfMissPredictions = np.sum(predicted != res)

    print(count)
    return w
def plotPoints(d1, d2, res, start, stop):

    d5 = d1[np.where(res > 0)]
    d6 = d2[np.where(res > 0)]
    d7 = d1[np.where(res < 0)]
    d8 = d2[np.where(res < 0)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.scatter(d5, d6, s=20, c='b', marker="s", label='positive')
    ax.scatter(d7, d8, s=20, c='r', marker="o", label='negative')
    m = -(-2/2)/(-2/1)  
    c = 2/2  
    a = np.arange(start-1, stop+1)
    f = a*m + c
    ax.plot(a,f)
    plt.legend(loc='upper right');
    plt.title('PLA')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()
d1, d2 = randomNumbers(-6, 8, 48)
res = function(d1,d2)
plotPoints(d1,d2,res,-6,8)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




