#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install surprise')


# In[2]:


import os
os.getcwd()


# In[3]:


import pandas as pd
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [10, 8]
import math
import seaborn as sns

np.random.seed(0)
#test

# In[4]:



#import the required libraries
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


electronics_df_full = pd.read_csv('Electronic_dataset.csv')


# In[6]:


electronics_df=electronics_df_full.head(n=20000)


# In[7]:


electronics_df.head()


# In[8]:


electronics_df.info()


# In[9]:


electronics_df.shape


# In[10]:


electronics_df.drop('timestamp',axis=1,inplace=True)


# In[11]:


electronics_df.info()


# In[12]:


rows,columns=electronics_df.shape
print('Number of rows: ',rows)
print('Number of columns: ',columns)


# In[13]:


electronics_df.dtypes


# In[14]:


#Taking subset of the dataset
electronics_df1=electronics_df.iloc[:20000,0:]


# In[15]:



electronics_df1.info()


# In[16]:



#Summary statistics of rating variable
electronics_df1['ratings'].describe().transpose()


# In[17]:


print('Minimum rating is: %d' %(electronics_df1.ratings.min()))
print('Maximum rating is: %d' %(electronics_df1.ratings.max()))


# 
# Ratings

# In[18]:


# Check the distribution of the rating
with sns.axes_style('white'):
    g = sns.factorplot("ratings", data=electronics_df1, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")


# 
# Handling Missing values

# In[19]:


#Check for missing values
print('Number of missing values across columns: \n',electronics_df.isnull().sum())


# 
# Users and products

# In[20]:


# Number of unique user id  in the data
print('Number of unique users in Raw data = ', electronics_df1['user_id'].nunique())
# Number of unique product id  in the data
print('Number of unique product in Raw data = ', electronics_df1['prod_id'].nunique())


# Taking the subset of dataset to make it less sparse/ denser

# In[21]:


#Check the top 10 users based on ratings
most_rated=electronics_df1.groupby('user_id').size().sort_values(ascending=False)[:10]
print('Top 10 users based on ratings: \n',most_rated)


# In[ ]:




