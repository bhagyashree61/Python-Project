#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[74]:


#importing the csv file
df=pd.read_csv(r"C:\Users\bhagy\Desktop\Wine prediction.csv")


# In[75]:


df.head()


# In[76]:


#checking the missing values
df.isnull().sum()


# In[77]:


#checking the no. of rows and columns
df.shape


#                  Data Analysis

# In[78]:


df.describe()


# In[79]:


sns.histplot(x='quality',data=df,edgecolor='Red')


# In[80]:


sns.barplot(x='quality',y='volatile acidity',data=df)


# In[81]:


# Quality Vs Citric acid
sns.barplot(x='quality',y='citric acid',data=df)


# In[82]:


#Data Preprocessing
x=df.drop('quality',axis=1)
y=df['quality']


# In[83]:


x.head()


# In[84]:


y.head()


# In[85]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[86]:


#Splitting the data into Train and Test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[87]:


x_train


# In[88]:


x_test


# In[89]:


y_train


# In[90]:


y_test


# In[91]:


#Model Training using Linear Regression
lr=LinearRegression()


# In[92]:


lr


# In[93]:


lr.fit(x_train,y_train)


# In[94]:


#Predicting the Values and comparing it with the actual values
y_pred=lr.predict(x_test)


# In[95]:


y_test.head()


# In[96]:


y_pred[0:5]


# In[97]:


# Calculate evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# In[ ]:




