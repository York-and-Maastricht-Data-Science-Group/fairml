#!/usr/bin/env python
# coding: utf-8

# # Sklearn compatible Exponentiated Gradient Reduction
# 
# Exponentiated gradient reduction is an in-processing technique that reduces fair classification to a sequence of cost-sensitive classification problems, returning a randomized classifier with the lowest empirical error subject to 
# fair classification constraints. The code for exponentiated gradient reduction wraps the source class 
# `fairlearn.reductions.ExponentiatedGradient` available in the https://github.com/fairlearn/fairlearn library,
# licensed under the MIT Licencse, Copyright Microsoft Corporation.

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

from aif360.sklearn.inprocessing import ExponentiatedGradientReduction

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


# We use Pandas for one-hot encoding for easy reference to columns associated with protected attributes, information necessary for Exponentiated Gradient Reduction

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


lr_aoe_sex = average_odds_error(y_test, y_pred, prot_attr='sex')
print(lr_aoe_sex)


# In[11]:


lr_aoe_race = average_odds_error(y_test, y_pred, prot_attr='race')
print(lr_aoe_race)


# ### Exponentiated Gradient Reduction

# Choose a base model for the randomized classifier

# In[12]:


estimator = LogisticRegression(solver='lbfgs')


# Determine the columns associated with the protected attribute(s)

# In[13]:


prot_attr_cols = [colname for colname in X_train if "sex" in colname or "race" in colname]


# Train the randomized classifier and observe test accuracy. Other options for `constraints` include "DemographicParity", "TruePositiveRateDifference," and "ErrorRateRatio."

# In[14]:


np.random.seed(0) #for reproducibility
exp_grad_red = ExponentiatedGradientReduction(prot_attr=prot_attr_cols, 
                                              estimator=estimator, 
                                              constraints="EqualizedOdds",
                                              drop_prot_attr=False)
exp_grad_red.fit(X_train, y_train)
egr_acc = exp_grad_red.score(X_test, y_test)
print(egr_acc)

# Check for that accuracy is comparable
assert abs(lr_acc-egr_acc)<=0.03


# In[15]:


egr_aoe_sex = average_odds_error(y_test, exp_grad_red.predict(X_test), prot_attr='sex')
print(egr_aoe_sex)

# Check for improvement in average odds error for sex
assert egr_aoe_sex<lr_aoe_sex


# In[16]:


egr_aoe_race = average_odds_error(y_test, exp_grad_red.predict(X_test), prot_attr='race')
print(egr_aoe_race)

# Check for improvement in average odds error for race
# assert egr_aoe_race<lr_aoe_race


# Number of calls made to base model algorithm

# In[17]:


exp_grad_red.model._n_oracle_calls


# Maximum calls permitted

# In[18]:


exp_grad_red.T


# Instead of passing in a value for `constraints`, we can also pass a `fairlearn.reductions.moment` object in for `constraints_moment`. You could use a predefined moment as we do below or create a custom moment using the fairlearn library.

# In[19]:


import fairlearn.reductions as red 

np.random.seed(0) #need for reproducibility
exp_grad_red2 = ExponentiatedGradientReduction(prot_attr=prot_attr_cols, 
                                              estimator=estimator, 
                                              constraints=red.EqualizedOdds(),
                                              drop_prot_attr=False)
exp_grad_red2.fit(X_train, y_train)
exp_grad_red2.score(X_test, y_test)


# In[20]:


average_odds_error(y_test, exp_grad_red2.predict(X_test), prot_attr='sex')


# In[21]:


average_odds_error(y_test, exp_grad_red2.predict(X_test), prot_attr='race')

