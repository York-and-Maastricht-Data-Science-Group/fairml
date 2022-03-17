#!/usr/bin/env python
# coding: utf-8

# #  Exponentiated Gradient Reduction
# 
# Exponentiated gradient reduction is an in-processing technique that reduces fair classification to a sequence of cost-sensitive classification problems, returning a randomized classifier with the lowest empirical error subject to 
# fair classification constraints. The code for exponentiated gradient reduction wraps the source class 
# `fairlearn.reductions.ExponentiatedGradient` available in the https://github.com/fairlearn/fairlearn library,
# licensed under the MIT Licencse, Copyright Microsoft Corporation.
# 
# This version of exponentiated gradient reduction (implemented in `aif360.algorithms`) wraps the sklearn compatible version of exponentiated gradient reduction implemented in `aif360.sklearn`. For a detailed tutorial on sklearn compatible exponentiated gradient reduction see [examples/sklearn/demo_exponentiated_gradient_reduction_sklearn.ipynb](sklearn/demo_exponentiated_gradient_reduction_sklearn.ipynb). 

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# Load all necessary packages
import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(0)


# #### Load dataset and set options

# In[2]:


# Get the dataset and split into train and test
dataset_orig = load_preproc_data_adult()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


# In[3]:


# print out some labels, names, etc.
display(Markdown("#### Training Dataset shape"))
print(dataset_orig_train.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_orig_train.privileged_protected_attributes, 
      dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)


# #### Metric for original training data

# In[4]:


# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())


# In[5]:


min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Scaled dataset - Verify that the scaling does not affect the group label statistics"))
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())


# ### Standard Logistic Regression

# In[6]:


X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()

lmod = LogisticRegression(solver='lbfgs')
lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)

X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()

y_pred = lmod.predict(X_test)

display(Markdown("#### Accuracy"))
lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# In[7]:


dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
dataset_orig_test_pred.labels = y_pred

# positive class index
pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

metric_test = ClassificationMetric(dataset_orig_test, 
                                    dataset_orig_test_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)
display(Markdown("#### Average odds difference"))
lr_aod = metric_test.average_odds_difference()
print(lr_aod)


# ### Exponentiated Gradient Reduction

# Choose a base model for the randomized classifer

# In[8]:


np.random.seed(0) #need for reproducibility
estimator = LogisticRegression(solver='lbfgs')


# Train the randomized classifier and observe test accuracy. Other options for `constraints` include "DemographicParity," "TruePositiveRateDifference", and "ErrorRateRatio."

# In[9]:



exp_grad_red = ExponentiatedGradientReduction(estimator=estimator, 
                                              constraints="EqualizedOdds",
                                              drop_prot_attr=False)
exp_grad_red.fit(dataset_orig_train)
exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)


# In[10]:


metric_test = ClassificationMetric(dataset_orig_test, 
                                   exp_grad_red_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

display(Markdown("#### Accuracy"))
egr_acc = metric_test.accuracy()
print(egr_acc)

#Check if accuracy is comparable
assert abs(lr_acc-egr_acc)<0.03

display(Markdown("#### Average odds difference"))
egr_aod = metric_test.average_odds_difference()
print(egr_aod)

#Check if average odds difference has improved
assert abs(egr_aod)<abs(lr_aod)

