#!/usr/bin/env python
# coding: utf-8

# # Sklearn compatible Grid Search for classification
# 
# Grid search is an in-processing technique that can be used for fair classification or fair regression. For classification it reduces fair classification to a sequence of cost-sensitive classification problems, returning the deterministic classifier with the lowest empirical error subject to fair classification constraints among
# the candidates searched. The code for grid search wraps the source class `fairlearn.reductions.GridSearch` available in the https://github.com/fairlearn/fairlearn library, licensed under the MIT Licencse, Copyright Microsoft Corporation.

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

from aif360.sklearn.inprocessing import GridSearchReduction

from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference


# ### Loading data

# Datasets are formatted as separate `X` (# samples x # features) and `y` (# samples x # labels) DataFrames. The index of each DataFrame contains protected attribute values per sample. Datasets may also load a `sample_weight` object to be used with certain algorithms/metrics. All of this makes it so that aif360 is compatible with scikit-learn objects.
# 
# For example, we can easily load the Adult dataset from UCI with the following line:

# In[2]:


X, y, sample_weight = fetch_adult()
X.head()


# In[3]:


# there is one unused category ('Never-worked') that was dropped during dropna
X.workclass.cat.remove_unused_categories(inplace=True)


# We can then map the protected attributes to integers,

# In[4]:


X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)


# and the target classes to 0/1,

# In[5]:


y = pd.Series(y.factorize(sort=True)[0], index=y.index)


# split the dataset,

# In[6]:


(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)


# We use Pandas for one-hot encoding for easy reference to columns associated with protected attributes, information necessary for grid search reduction.

# In[7]:


X_train, X_test = pd.get_dummies(X_train), pd.get_dummies(X_test)
X_train.head()


# The protected attribute information is also replicated in the labels:

# In[8]:


y_train.head()


# ### Running metrics

# With the data in this format, we can easily train a scikit-learn model and get predictions for the test data:

# In[9]:


y_pred = LogisticRegression(solver='lbfgs').fit(X_train, y_train).predict(X_test)
lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# We can assess how close the predictions are to equality of odds.
# 
# `average_odds_error()` computes the (unweighted) average of the absolute values of the true positive rate (TPR) difference and false positive rate (FPR) difference, i.e.:
# 
# $$ \tfrac{1}{2}\left(|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}| + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|\right) $$

# In[10]:


lr_aoe = average_odds_error(y_test, y_pred, prot_attr='sex')
print(lr_aoe)


# ### Grid Search

# Choose a base model for the candidate classifiers. Base models should implement a fit method that can take a sample weight as input. For details refer to the docs. 

# In[11]:


estimator = LogisticRegression(solver='lbfgs')


# Determine the columns associated with the protected attribute(s). Grid search can handle more then one attribute but it is computationally expensive. A similar method with less computational overhead is exponentiated gradient reduction, detailed at [examples/sklearn/demo_exponentiated_gradient_reduction_sklearn.ipynb](sklearn/demo_exponentiated_gradient_reduction_sklearn.ipynb).

# In[12]:


prot_attr_cols = [colname for colname in X_train if "sex" in colname]


# Search for the best classifier and observe test accuracy. Other options for `constraints` include "DemographicParity," "TruePositiveRateDifference", and "ErrorRateRatio."

# In[13]:


np.random.seed(0) #need for reproducibility
grid_search_red = GridSearchReduction(prot_attr=prot_attr_cols, 
                                      estimator=estimator, 
                                      constraints="EqualizedOdds",
                                      grid_size=20,
                                      drop_prot_attr=False)
grid_search_red.fit(X_train, y_train)
gs_acc = grid_search_red.score(X_test, y_test)
print(gs_acc)

#Check if accuracy is comparable
assert abs(lr_acc-gs_acc)<0.03


# In[14]:


gs_aoe = average_odds_error(y_test, grid_search_red.predict(X_test), prot_attr='sex')
print(gs_aoe)

#Check if average odds error improved
assert gs_aoe<lr_aoe


# Instead of passing in a value for `constraints`, we can also pass a `fairlearn.reductions.moment` object in for `constraints_moment`. You could use a predefined moment as we do below or create a custom moment using the fairlearn library.

# In[15]:


import fairlearn.reductions as red 

np.random.seed(0) #need for reproducibility
grid_search_red = GridSearchReduction(prot_attr=prot_attr_cols, 
                                      estimator=estimator, 
                                      constraints=red.EqualizedOdds(),
                                      grid_size=20,
                                      drop_prot_attr=False)
grid_search_red.fit(X_train, y_train)
grid_search_red.score(X_test, y_test)


# In[16]:


average_odds_error(y_test, grid_search_red.predict(X_test), prot_attr='sex')

