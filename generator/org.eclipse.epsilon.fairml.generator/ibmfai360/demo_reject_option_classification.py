#!/usr/bin/env python
# coding: utf-8

# #### This notebook demonstrates the use of the Reject Option Classification (ROC) post-processing algorithm for bias mitigation.
# - The debiasing function used is implemented in the `RejectOptionClassification` class.
# - Divide the dataset into training, validation, and testing partitions.
# - Train classifier on original training data.
# - Estimate the optimal classification threshold, that maximizes balanced accuracy without fairness constraints.
# - Estimate the optimal classification threshold, and the critical region boundary (ROC margin) using a validation set for the desired constraint on fairness. The best parameters are those that maximize the classification threshold while satisfying the fairness constraints.
# - The constraints can be used on the following fairness measures:
#     * Statistical parity difference on the predictions of the classifier
#     * Average odds difference for the classifier
#     * Equal opportunity difference for the classifier
# - Determine the prediction scores for testing data. Using the estimated optimal classification threshold, compute accuracy and fairness metrics.
# - Using the determined optimal classification threshold and the ROC margin, adjust the predictions. Report accuracy and fairness metric on the new predictions.

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# Load all necessary packages
import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from warnings import warn

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.postprocessing.reject_option_classification        import RejectOptionClassification
from common_utils import compute_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider


# #### Load dataset and specify options

# In[2]:


## import dataset
dataset_used = "adult" # "adult", "german", "compas"
protected_attribute_used = 1 # 1, 2

if dataset_used == "adult":
#     dataset_orig = AdultDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_adult(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_adult(['race'])
    
elif dataset_used == "german":
#     dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_german(['sex'])
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = load_preproc_data_german(['age'])
    
elif dataset_used == "compas":
#     dataset_orig = CompasDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_compas(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]  
        dataset_orig = load_preproc_data_compas(['race'])

        
# Metric used (should be one of allowed_metrics)
metric_name = "Statistical parity difference"

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05
        
#random seed for calibrated equal odds prediction
np.random.seed(1)

# Verify metric name
allowed_metrics = ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]
if metric_name not in allowed_metrics:
    raise ValueError("Metric name should be one of allowed metrics")


# #### Split into train, test and validation

# In[3]:


# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)


# #### Clean up training data and display properties of the data

# In[4]:


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

# In[5]:


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# ### Train classifier on original data

# In[6]:


# Logistic regression classifier and predictions
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train)
y_train_pred = lmod.predict(X_train)

# positive class index
pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
dataset_orig_train_pred.labels = y_train_pred


# #### Obtain scores for validation and test sets

# In[7]:


dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
y_valid = dataset_orig_valid_pred.labels
dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_orig.transform(dataset_orig_test_pred.features)
y_test = dataset_orig_test_pred.labels
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)


# ### Find the optimal parameters from the validation set

# #### Best threshold for classification only (no fairness)

# In[8]:


num_thresh = 100
ba_arr = np.zeros(num_thresh)
class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
for idx, class_thresh in enumerate(class_thresh_arr):
    
    fav_inds = dataset_orig_valid_pred.scores > class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    
    classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                             dataset_orig_valid_pred, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()                       +classified_metric_orig_valid.true_negative_rate())

best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
best_class_thresh = class_thresh_arr[best_ind]

print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)


# #### Estimate optimal parameters for the ROC method

# In[9]:


ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name=metric_name,
                                  metric_ub=metric_ub, metric_lb=metric_lb)
ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)


# In[10]:


print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
print("Optimal ROC margin = %.4f" % ROC.ROC_margin)


# ### Predictions from Validation Set

# In[11]:


# Metrics for the test set
fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

display(Markdown("#### Validation set"))
display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))

metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred, 
                unprivileged_groups, privileged_groups)


# In[12]:


# Transform the validation set
dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)

display(Markdown("#### Validation set"))
display(Markdown("##### Transformed predictions - With fairness constraints"))
metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred, 
                unprivileged_groups, privileged_groups)


# In[13]:


# Testing: Check if the metric optimized has not become worse
assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])


# ### Predictions from Test Set

# In[14]:


# Metrics for the test set
fav_inds = dataset_orig_test_pred.scores > best_class_thresh
dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

display(Markdown("#### Test set"))
display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))

metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                unprivileged_groups, privileged_groups)


# In[15]:


# Metrics for the transformed test set
dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

display(Markdown("#### Test set"))
display(Markdown("##### Transformed predictions - With fairness constraints"))
metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                unprivileged_groups, privileged_groups)


# # Summary of Optimal Parameters
# We show the optimal parameters for all combinations of metrics optimized, datasets, and protected attributes below.

# ### Fairness Metric: Statistical parity difference, Accuracy Metric: Balanced accuracy
# 
# #### Performance
# 
# | Dataset |Sex (Acc-Bef)|Sex (Acc-Aft)|Sex (Fair-Bef)|Sex (Fair-Aft)|Race/Age (Acc-Bef)|Race/Age (Acc-Aft)|Race/Age (Fair-Bef)|Race/Age (Fair-Aft)|
# |-|-|-|-|-|-|-|-|-|
# |Adult (Valid)|0.7473|0.6051|-0.3703|-0.0436|0.7473|0.6198|-0.2226|-0.0007|
# |Adult (Test)|0.7417|0.5968|-0.3576|-0.0340|0.7417|0.6202|-0.2279|0.0006|
# |German (Valid)|0.6930|0.6991|-0.0613|0.0429|0.6930|0.6607|-0.2525|-0.0328|
# |German (Test)|0.6524|0.6460|-0.0025|0.0410|0.6524|0.6317|-0.3231|-0.1038|
# |Compas (Valid)|0.6599|0.6400|-0.2802|0.0234|0.6599|0.6646|-0.3225|-0.0471|
# |Compas (Test)|0.6774|0.6746|-0.2724|-0.0313|0.6774|0.6512|-0.2494|0.0578|
# 
# #### Optimal Parameters
# 
# | Dataset |Sex (Class. thresh.)|Sex (Class. thresh. - fairness)|Sex (ROC margin - fairness)| Race/Age (Class. thresh.)|Race/Age (Class. thresh. - fairness)|Race/Age (ROC margin - fairness)|
# |-|-|-|-|-|-|-|
# |Adult|0.2674|0.5049|0.1819|0.2674|0.5049|0.0808|
# |German|0.6732|0.6237|0.0538|0.6732|0.7029|0.0728|
# |Compas|0.5148|0.5841|0.0679|0.5148|0.5841|0.0679|

# ### Fairness Metric: Average odds difference, Accuracy Metric: Balanced accuracy
# 
# #### Performance
# 
# | Dataset |Sex (Acc-Bef)|Sex (Acc-Aft)|Sex (Fair-Bef)|Sex (Fair-Aft)|Race/Age (Acc-Bef)|Race/Age (Acc-Aft)|Race/Age (Fair-Bef)|Race/Age (Fair-Aft)|
# |-|-|-|-|-|-|-|-|-|
# |Adult (Valid)|0.7473|0.6058|-0.2910|-0.0385|0.7473|0.6593|-0.1947|-0.0444|
# |Adult (Test)|0.7417|0.6024|-0.3281|-0.0438|0.7417|0.6611|-0.1991|-0.0121|
# |German (Valid)|0.6930|0.6930|-0.0039|-0.0039|0.6930|0.6807|-0.0919|-0.0193|
# |German (Test)|0.6524|0.6571|0.0071|0.0237|0.6524|0.6587|-0.3278|-0.2708|
# |Compas (Valid)|0.6599|0.6416|-0.2285|-0.0332|0.6599|0.6646|-0.2918|-0.0105|
# |Compas (Test)|0.6774|0.6721|-0.2439|-0.0716|0.6774|0.6512|-0.1927|0.1145|
# 
# #### Optimal Parameters
# 
# | Dataset |Sex (Class. thresh.)|Sex (Class. thresh. - fairness)|Sex (ROC margin - fairness)| Race/Age (Class. thresh.)|Race/Age (Class. thresh. - fairness)|Race/Age (ROC margin - fairness)|
# |-|-|-|-|-|-|-|
# |Adult|0.2674|0.5049|0.1212|0.2674|0.5049|0.0505|
# |German|0.6732|0.6633|0.0137|0.6732|0.6732|0.0467|
# |Compas|0.5148|0.5742|0.0608|0.5148|0.5841|0.0679|
# 

# ### Fairness Metric: Equal opportunity difference, Accuracy Metric: Balanced accuracy
# 
# #### Performance
# 
# | Dataset |Sex (Acc-Bef)|Sex (Acc-Aft)|Sex (Fair-Bef)|Sex (Fair-Aft)|Race/Age (Acc-Bef)|Race/Age (Acc-Aft)|Race/Age (Fair-Bef)|Race/Age (Fair-Aft)|
# |-|-|-|-|-|-|-|-|-|
# |Adult (Valid)|0.7473|0.6051|-0.3066|-0.0136|0.7473|0.6198|-0.2285|0.0287|
# |Adult (Test)|0.7417|0.5968|-0.4001|-0.0415|0.7417|0.6202|-0.2165|0.1193|
# |German (Valid)|0.6930|0.6930|-0.0347|-0.0347|0.6930|0.6597|0.1162|-0.0210|
# |German (Test)|0.6524|0.6571|0.0400|0.0733|0.6524|0.6190|-0.3556|-0.4333|
# |Compas (Valid)|0.6599|0.6416|-0.1938|0.0244|0.6599|0.6646|-0.2315|0.0002|
# |Compas (Test)|0.6774|0.6721|-0.1392|0.0236|0.6774|0.6512|-0.1877|0.1196|
# 
# #### Optimal Parameters
# 
# | Dataset |Sex (Class. thresh.)|Sex (Class. thresh. - fairness)|Sex (ROC margin - fairness)| Race/Age (Class. thresh.)|Race/Age (Class. thresh. - fairness)|Race/Age (ROC margin - fairness)|
# |-|-|-|-|-|-|-|
# |Adult|0.2674|0.5049|0.1819|0.2674|0.5049|0.0808|
# |German|0.6732|0.6633|0.0137|0.6732|0.6039|0.0000|
# |Compas|0.5148|0.5742|0.0608|0.5148|0.5841|0.0679|
# 

# In[ ]:




