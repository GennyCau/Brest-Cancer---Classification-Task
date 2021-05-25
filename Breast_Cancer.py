#!/usr/bin/env python
# coding: utf-8

# # Hello everyone! 
# ### Here i'm predicting whether a cancer is benign or malignant based on Brest Cancer Winsconsin Data Set from Kaggle (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

# #### The Dataset contains the following informations:
# 
# - ID number
# - Diagnosis (M = malignant, B = benign)
# - Ten real-valued features are computed for each cell nucleus:
# 
#    - radius 
#    - texture
#    - perimeter
#    - area
#    - smoothness 
#    - compactness
#    - concavity
#    - concave points
#    - symmetry
#    - fractal dimension

# ###### Here you can find the libreries used for the analysis and a preview of the dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


pd.set_option("display.max_columns", None)

df = pd.read_csv('breastcancer.csv')
df.head()


# ##### I noticed that the column named 'Unnamed: 32' contains only NaN inputs

# In[3]:


df['Unnamed: 32'].value_counts()


# ##### I decided to remove both 'Unnamed: 32' and 'id' (which contains the identification code of the patient) columns because these are not significative for the analysis 

# In[4]:


df.drop(columns=['Unnamed: 32', 'id'], inplace=True)


# ###### Now none of the data is NaN

# In[5]:


df.isnull().sum()


# ###### The 'diagnosis' column contains the classes, while all the other features are continuous and have float data type

# In[6]:


df.info()


# ##### Here the statistics of the continuous features

# In[7]:


df.describe().T


# ##### The dataset is unbalanced: the majority of the inputs are classified as benign

# In[8]:


sns.countplot(x='diagnosis', data=df)
plt.title('Data distribution between classes')
plt.show()


# ##### In order to perform the analysis, the label must be encoded

# In[9]:


le = LabelEncoder()


# In[10]:


df['diagnosis'] = le.fit_transform(df['diagnosis'])


# #### Here I prepared the data for the analysis. 
# ###### For the split, I used the stratify option in order to counter the unbalanceness of the classes.

# In[11]:


X = df.drop(columns='diagnosis')


# In[12]:


y = df['diagnosis']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ##### For this classification task I decided to use the classifier based on Random Forest algorithm

# In[14]:


rm = RandomForestClassifier(random_state=42)


# In[15]:


rm.fit(X_train, y_train)


# ##### It is interesting to visualize the feature importances for this classification problem

# In[16]:


def feature_importances_plot(model, labels, **kwargs):
    """
    Compute normalized feature importance from model
    and return the data or show plot.
    
    Parameters
    ----------
    model : any
        scikit-learn model
    labels : list | np.array
        list of feature labels
    
    Returns
    -------
    AxesSubplot
    """
    feature_importances = model.feature_importances_
    feature_importances = 100 * (feature_importances / feature_importances.max())
    df = pd.DataFrame(data={"feature_importances": feature_importances}, index=labels)
    df.sort_values(by="feature_importances", inplace=True)
    return df.plot(kind="barh", figsize=(8, 10), title=f"Feature importances", legend=None, **kwargs)


# In[17]:


feature_importances_plot(model=rm, labels=X.columns)
plt.show()


# ##### In the end, let's calculate the accuracy on the test set

# In[18]:


rm.score(X_test, y_test)


# ##### The accuracy obtained on the test set is high, we can conclude that Random Forest is a good classifier for this problem.

# ## I hope you find this analysis interesting!
# 
# ## Thank you for reading!
