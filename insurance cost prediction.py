#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


df=pd.read_csv(r"C:\Users\bhagy\Desktop\Medical insurence.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


sns.histplot(x='age',data=df,edgecolor='Red')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[8]:


df.columns


# In[9]:


sns.countplot(x='sex',data=df)
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('frequency')


# In[10]:


sns.histplot(x='bmi',data=df)
plt.title('BMI')
plt.xlabel('bmi')
plt.ylabel('frequency')


# In[11]:


sns.countplot(x='smoker',data=df)
plt.title('Smoker')
plt.xlabel('smoker')
plt.ylabel('frequency')


# In[12]:


sns.histplot(x='charges',data=df)


# In[13]:


sns.histplot(x='region',data=df)


# In[14]:


df.replace({'sex':{'male':0,'female':1}},inplace=True)
df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# In[15]:


df.head()


# In[16]:


x=df.drop('charges',axis=1)
y=df['charges']


# In[17]:


x


# In[18]:


y


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=2)


# In[20]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[21]:


training_data_prediction=regressor.predict(x_train)


# In[22]:


r2_train=metrics.r2_score(y_train,training_data_prediction)
print(r2_train)


# In[23]:


df.tail()


# In[24]:


input_data=(21,0,36.,1,0,3)
data_array=np.asarray(input_data)
reshaped_array=data_array.reshape(1,-1)
prediction=regressor.predict(reshaped_array)
prediction


# In[ ]:





# In[ ]:




