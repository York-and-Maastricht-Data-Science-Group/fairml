#!/usr/bin/env python
# coding: utf-8

# # Sklearn compatible Grid Search for regression
# 
# Grid search is an in-processing technique that can be used for fair classification or fair regression. For classification it reduces fair classification to a sequence of cost-sensitive classification problems, returning the deterministic classifier with the lowest empirical error subject to fair classification constraints among
# the candidates searched. For regression it uses the same priniciple to return a deterministic regressor with the lowest empirical error subject to the constraint of bounded group loss. The code for grid search wraps the source class `fairlearn.reductions.GridSearch` available in the https://github.com/fairlearn/fairlearn library, licensed under the MIT Licencse, Copyright Microsoft Corporation.

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from aif360.sklearn.inprocessing import GridSearchReduction

from aif360.sklearn.datasets import fetch_lawschool_gpa


# ### Loading data

# Datasets are formatted as separate `X` (# samples x # features) and `y` (# samples x # labels) DataFrames. The index of each DataFrame contains protected attribute values per sample. Datasets may also load a `sample_weight` object to be used with certain algorithms/metrics. All of this makes it so that aif360 is compatible with scikit-learn objects.
# 
# For example, we can easily load the law school gpa dataset from tempeh with the following line:

# In[2]:


X_train, y_train = fetch_lawschool_gpa(subset="train")
X_test, y_test = fetch_lawschool_gpa(subset="test")
X_train.head()


# We can then map the protected attributes to integers,

# In[3]:


X_train.index = pd.MultiIndex.from_arrays(X_train.index.codes, names=X_train.index.names)
X_test.index = pd.MultiIndex.from_arrays(X_test.index.codes, names=X_test.index.names)
y_train.index = pd.MultiIndex.from_arrays(y_train.index.codes, names=y_train.index.names)
y_test.index = pd.MultiIndex.from_arrays(y_test.index.codes, names=y_test.index.names)


# We use Pandas for one-hot encoding for easy reference to columns associated with protected attributes, information necessary for grid search reduction.

# In[4]:


X_train, X_test = pd.get_dummies(X_train), pd.get_dummies(X_test)
X_train.head()


# We normalize the continuous values

# In[5]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train.values),columns=list(X_train),index=X_train.index)
X_test = pd.DataFrame(min_max_scaler.transform(X_test.values),columns=list(X_test),index=X_test.index)
X_train.head()


# In[6]:


min_max_scaler = preprocessing.MinMaxScaler()
y_train = pd.Series(min_max_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(),index=y_train.index)
y_test = pd.Series(min_max_scaler.transform(y_test.values.reshape(-1, 1)).flatten(),index=y_test.index)


# The protected attribute information is also replicated in the labels:

# In[7]:


y_train.head()


# ### Running metrics

# With the data in this format, we can easily train a scikit-learn model and get predictions for the test data. We drop the protective attribule columns so that they are not used in the model.

# In[8]:


prot_attr_cols = [col for col in list(X_train) if "race" in col]


# In[9]:


lr = LinearRegression().fit(X_train.drop(prot_attr_cols,axis=1), y_train)
y_pred = lr.predict(X_test.drop(prot_attr_cols, axis=1))
lr_mae = mean_absolute_error(y_test, y_pred)
print(lr_mae)


# We can assess how the mean absolute error differs across groups

# In[10]:


X_test_white = X_test.iloc[X_test.index.get_level_values('race') == 1]
y_test_white = y_test.iloc[y_test.index.get_level_values('race') == 1]

y_pred_white = lr.predict(X_test_white.drop(prot_attr_cols, axis=1))

lr_mae_w = mean_absolute_error(y_test_white, y_pred_white)
print("White:", lr_mae_w)


# In[11]:


X_test_black = X_test.iloc[X_test.index.get_level_values('race') == 0]
y_test_black = y_test.iloc[y_test.index.get_level_values('race') == 0]

y_pred_black = lr.predict(X_test_black.drop(prot_attr_cols, axis=1))

lr_mae_b = mean_absolute_error(y_test_black, y_pred_black)
print("Black:", lr_mae_b)


# In[12]:


print("Mean absolute error difference across groups:", lr_mae_b-lr_mae_w)


# ### Grid Search

# Choose a base model for the candidate regressors. Base models should implement a fit method that can take a sample weight as input. For details refer to the docs. 

# In[13]:


estimator = LinearRegression()


# Search for the best regressor and observe mean absolute error. Grid search for regression uses "GroupLoss" to specify using bounded group loss for its constraints. Accordingly we need to specify a loss function, like "Absolute." Other options include "Square" and "ZeroOne." When the loss is "Absolute" or "Square" we also specify the expected range of the y values in min_val and max_val. For details on the implementation of these loss function see the fairlearn library here https://github.com/fairlearn/fairlearn/blob/master/fairlearn/reductions/_moments/bounded_group_loss.py.

# In[14]:


np.random.seed(0) #need for reproducibility
grid_search_red = GridSearchReduction(prot_attr=prot_attr_cols, 
                                      estimator=estimator, 
                                      constraints="GroupLoss",
                                      loss="Absolute",
                                      min_val=0,
                                      max_val=1,
                                      grid_size=10,
                                      drop_prot_attr=True)
grid_search_red.fit(X_train, y_train)
gs_pred = grid_search_red.predict(X_test)
gs_mae = mean_absolute_error(y_test, gs_pred)
print(gs_mae)

#Check if mean absolute error is comparable
assert abs(gs_mae-lr_mae)<0.01


# In[15]:


gs_mae_w = mean_absolute_error(y_test_white, grid_search_red.predict(X_test_white))
print("White:", gs_mae_w)


# In[16]:


gs_mae_b = mean_absolute_error(y_test_black, grid_search_red.predict(X_test_black))
print("Black:", gs_mae_b)


# In[17]:


print("Mean absolute error difference across groups:", gs_mae_b-gs_mae_w)

#Check if difference decreased
assert (gs_mae_b-gs_mae_w)<(lr_mae_b-lr_mae_w)

