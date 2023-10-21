#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px


# In[3]:


#reading the csv file
df=pd.read_csv(r"C:\Users\bhagy\Desktop\titanic.csv")


# In[4]:


df.head()


# In[5]:


#total rows & columns
df.shape


# In[10]:


#getting to know the type of data
df.info()


# In[11]:


#describing the data
df.describe()


# DATA CLEANING

# In[12]:


#Finding the null values
df.isnull().sum()


# In[20]:


#finding the duplicate
df[df.duplicated()]


# In[21]:


#dropping the 'cabin' column as therewer many missing values
df.drop(['Cabin'],axis=1,inplace=True)


# In[24]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[25]:


#checking as no null is left
df.isnull().sum()


# DATA ANALYSIS

# In[27]:


df.head()


# In[35]:


# People who survived during the incident
(df['Survived']== 1).sum()
print('the people who survived is:',(df['Survived']==1).sum())

(df['Survived']== 0).sum()
print('the people who died is:',(df['Survived']== 0).sum())


# In[55]:


sns.countplot(x='Survived',data=df)
plt.xlabel=('Number of people')
plt.ylabel=('No of people')


# In[58]:


df['Pclass'].unique()


# In[61]:


#Value count Pclass
df['Pclass'].value_counts()


# In[60]:


#count of Pclass
sns.countplot(x='Pclass',data=df)


# In[63]:


#Males age between 22 to 30
df[(df['Sex']=='male')&(df['Age'].between(22,30))]


# In[64]:


#count as per sex
sns.countplot(x=df['Sex'],data=df)


# In[65]:


sns.distplot(df['Age'],bins=5)


# In[72]:


#histogram showing the max age people in the ship
plt.hist(x="Age",data=df,edgecolor='green')


# In[79]:


#sex wise Survived
sns.barplot(x='Sex',y='Survived',data=df)


# In[81]:


#pclass who survived
sns.barplot(x='Pclass',y='Survived',data=df)


# In[82]:


df.groupby('Pclass')['Survived'].sum()


# In[85]:


#Pclass acording to age
sns.barplot(x='Pclass',y="Age",data=df,hue='Sex')


# In[88]:


#relating the emabared and Fare Relation according to sex
sns.boxplot(x='Embarked',y='Fare',hue='Sex',data=df,dodge=True)


# In[89]:



df['SibSp'].unique()


# In[90]:


#total siblings/Spouse
df['SibSp'].sum()


# In[91]:


#finding the Total Sibling/spouses across diff Pclass
df.groupby('Pclass')['SibSp'].sum()


# In[105]:


#survived number as per sex
df.groupby('Sex')['Survived'].sum()


# In[ ]:




