#!/usr/bin/env python
# coding: utf-8

# # Getting Started

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

from aif360.sklearn.preprocessing import ReweighingMeta
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference


# ## Loading data

# Datasets are formatted as separate `X` (# samples x # features) and `y` (# samples x # labels) DataFrames. The index of each DataFrame contains protected attribute values per sample. Datasets may also load a `sample_weight` object to be used with certain algorithms/metrics. All of this makes it so that aif360 is compatible with scikit-learn objects.
# 
# For example, we can easily load the Adult dataset from UCI with the following line:

# In[2]:


X, y, sample_weight = fetch_adult()
X.head()


# We can then map the protected attributes to integers,

# In[3]:


X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)


# and the target classes to 0/1,

# In[4]:


y = pd.Series(y.factorize(sort=True)[0], index=y.index)


# split the dataset,

# In[5]:


(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)


# and finally, one-hot encode the categorical features:

# In[6]:


ohe = make_column_transformer(
        (OneHotEncoder(sparse=False), X_train.dtypes == 'category'),
        remainder='passthrough')
X_train  = pd.DataFrame(ohe.fit_transform(X_train), index=X_train.index)
X_test = pd.DataFrame(ohe.transform(X_test), index=X_test.index)

X_train.head()


# Note: the column names are lost in this transformation. The same encoding can be done with Pandas, but this cannot be combined with other preprocessing in a Pipeline.

# In[7]:


# there is one unused category ('Never-worked') that was dropped during dropna
X.workclass.cat.remove_unused_categories(inplace=True)
pd.get_dummies(X).head()


# The protected attribute information is also replicated in the labels:

# In[8]:


y_train.head()


# ## Running metrics

# With the data in this format, we can easily train a scikit-learn model and get predictions for the test data:

# In[9]:


y_pred = LogisticRegression(solver='lbfgs').fit(X_train, y_train).predict(X_test)
accuracy_score(y_test, y_pred)


# Now, we can analyze our predictions and quickly calucate the disparate impact for females vs. males:

# In[10]:


disparate_impact_ratio(y_test, y_pred, prot_attr='sex')


# And similarly, we can assess how close the predictions are to equality of odds.
# 
# `average_odds_error()` computes the (unweighted) average of the absolute values of the true positive rate (TPR) difference and false positive rate (FPR) difference, i.e.:
# 
# $$ \tfrac{1}{2}\left(|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}| + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|\right) $$

# In[11]:


average_odds_error(y_test, y_pred, prot_attr='sex')


# ## Debiasing algorithms

# `ReweighingMeta` is a workaround until changing sample weights can be handled properly in `Pipeline`/`GridSearchCV`

# In[12]:


rew = ReweighingMeta(estimator=LogisticRegression(solver='lbfgs'))

params = {'estimator__C': [1, 10], 'reweigher__prot_attr': ['sex']}

clf = GridSearchCV(rew, params, scoring='accuracy', cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.best_params_)


# In[13]:


disparate_impact_ratio(y_test, clf.predict(X_test), prot_attr='sex')


# Rather than trying to weight accuracy and fairness, we can try a fair in-processing algorithm:

# In[14]:


adv_deb = AdversarialDebiasing(prot_attr='sex', random_state=1234567)
adv_deb.fit(X_train, y_train)
adv_deb.score(X_test, y_test)


# In[15]:


average_odds_error(y_test, adv_deb.predict(X_test), prot_attr='sex')


# Note that `AdversarialDebiasing` creates a TensorFlow session which we should close when we're finished to free up resources:

# In[16]:


adv_deb.sess_.close()


# Finally, let's try a post-processor, `CalibratedEqualizedOdds`.
# 
# Since the post-processor needs to be trained on data unseen by the original estimator, we will use the `PostProcessingMeta` class which splits the data and trains the estimator and post-processor with their own split.

# In[17]:


cal_eq_odds = CalibratedEqualizedOdds('sex', cost_constraint='fnr', random_state=1234567)
log_reg = LogisticRegression(solver='lbfgs')
postproc = PostProcessingMeta(estimator=log_reg, postprocessor=cal_eq_odds, random_state=1234567)

postproc.fit(X_train, y_train)
accuracy_score(y_test, postproc.predict(X_test))


# In[18]:


y_pred = postproc.predict_proba(X_test)[:, 1]
y_lr = postproc.estimator_.predict_proba(X_test)[:, 1]
br = postproc.postprocessor_.base_rates_
i = X_test.index.get_level_values('sex') == 1

plt.plot([0, br[0]], [0, 1-br[0]], '-b', label='All calibrated classifiers (Females)')
plt.plot([0, br[1]], [0, 1-br[1]], '-r', label='All calibrated classifiers (Males)')

plt.scatter(generalized_fpr(y_test[~i], y_lr[~i]),
            generalized_fnr(y_test[~i], y_lr[~i]),
            300, c='b', marker='.', label='Original classifier (Females)')
plt.scatter(generalized_fpr(y_test[i], y_lr[i]),
            generalized_fnr(y_test[i], y_lr[i]),
            300, c='r', marker='.', label='Original classifier (Males)')
                                                                        
plt.scatter(generalized_fpr(y_test[~i], y_pred[~i]),
            generalized_fnr(y_test[~i], y_pred[~i]),
            100, c='b', marker='d', label='Post-processed classifier (Females)')
plt.scatter(generalized_fpr(y_test[i], y_pred[i]),
            generalized_fnr(y_test[i], y_pred[i]),
            100, c='r', marker='d', label='Post-processed classifier (Males)')

plt.plot([0, 1], [generalized_fnr(y_test, y_pred)]*2, '--', c='0.5')

plt.axis('square')
plt.xlim([0.0, 0.4])
plt.ylim([0.3, 0.7])
plt.xlabel('generalized fpr');
plt.ylabel('generalized fnr');
plt.legend(bbox_to_anchor=(1.04,1), loc='upper left');


# We can see the generalized false negative rate is approximately equalized and the classifiers remain close to the calibration lines.
# 
# We can quanitify the discrepancy between protected groups using the `difference` operator:

# In[19]:


difference(generalized_fnr, y_test, y_pred, prot_attr='sex')

