#!/usr/bin/env python
# coding: utf-8

# # Meta-Algorithm for fair classification.
# The fairness metrics to be optimized have to specified as "input". Currently we can handle the following fairness metrics:
# Statistical Rate, False Positive Rate, True Positive Rate, False Negative Rate, True Negative Rate,
# Accuracy Rate, False Discovery Rate, False Omission Rate, Positive Predictive Rate, Negative Predictive Rate.
# 
# -----------------------------
# 
# The example below considers the cases of False Discovery Parity and Statistical Rate (disparate impact).
# 

# In[1]:


from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing import MetaFairClassifier

np.random.seed(12345)


# ## Original Training dataset

# In[2]:


dataset_orig = load_preproc_data_adult()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


# In[3]:


min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)


# In[4]:


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


# In[5]:


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = {:.3f}".format(metric_orig_train.mean_difference()))
metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {:.3f}".format(metric_orig_test.mean_difference()))


# ## Algorithm without debiasing
# 
# Get classifier without fairness constraints

# In[6]:


biased_model = MetaFairClassifier(tau=0, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)


# Apply the unconstrained model to test data

# In[7]:


dataset_bias_test = biased_model.predict(dataset_orig_test)


# In[8]:


classified_metric_bias_test = ClassificationMetric(dataset_orig_test, dataset_bias_test,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
print("Test set: Classification accuracy = {:.3f}".format(classified_metric_bias_test.accuracy()))
TPR = classified_metric_bias_test.true_positive_rate()
TNR = classified_metric_bias_test.true_negative_rate()
bal_acc_bias_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = {:.3f}".format(bal_acc_bias_test))
print("Test set: Disparate impact = {:.3f}".format(classified_metric_bias_test.disparate_impact()))
fdr = classified_metric_bias_test.false_discovery_rate_ratio()
fdr = min(fdr, 1/fdr)
print("Test set: False discovery rate ratio = {:.3f}".format(fdr))


# ## Debiasing with FDR objective
# 
# Learn a debiased classifier

# In[9]:


debiased_model = MetaFairClassifier(tau=0.7, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)


# Apply the debiased model to test data

# In[10]:


dataset_debiasing_test = debiased_model.predict(dataset_orig_test)


# ### Model - with debiasing - dataset metrics

# In[11]:


metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {:.3f}".format(metric_dataset_debiasing_test.mean_difference()))


# ### Model - with debiasing - classification metrics

# In[12]:


classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
                                                 dataset_debiasing_test,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
print("Test set: Classification accuracy = {:.3f}".format(classified_metric_debiasing_test.accuracy()))
TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = {:.3f}".format(bal_acc_debiasing_test))
print("Test set: Disparate impact = {:.3f}".format(classified_metric_debiasing_test.disparate_impact()))
fdr = classified_metric_debiasing_test.false_discovery_rate_ratio()
fdr = min(fdr, 1/fdr)
print("Test set: False discovery rate ratio = {:.3f}".format(fdr))


# We see that the FDR ratio has increased meaning it is now closer to parity.

# ## Running the algorithm for different tau values

# In[13]:


accuracies, statistical_rates = [], []
s_attr = "race"

all_tau = np.linspace(0, 0.9, 10)
for tau in tqdm(all_tau):
    debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=s_attr, type='sr')
    debiased_model.fit(dataset_orig_train)

    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    metric = ClassificationMetric(dataset_orig_test, dataset_debiasing_test,
                                  unprivileged_groups=[{s_attr: 0}],
                                  privileged_groups=[{s_attr: 1}])

    accuracies.append(metric.accuracy())
    sr = metric.disparate_impact()
    statistical_rates.append(min(sr, 1/sr))


# Output fairness is represented by $\gamma_{sr}$, which is the disparate impact ratio of different sensitive attribute values.

# In[14]:


fig, ax1 = plt.subplots(figsize=(13,7))
ax1.plot(all_tau, accuracies, color='r')
ax1.set_title('Accuracy and $\gamma_{sr}$ vs Tau', fontsize=16, fontweight='bold')
ax1.set_xlabel('Input Tau', fontsize=16, fontweight='bold')
ax1.set_ylabel('Accuracy', color='r', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(all_tau, statistical_rates, color='b')
ax2.set_ylabel('$\gamma_{sr}$', color='b', fontsize=16, fontweight='bold')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)


# References:
# 
#      Celis, L. E., Huang, L., Keswani, V., & Vishnoi, N. K. (2018). 
#      "Classification with Fairness Constraints: A Meta-Algorithm with Provable Guarantees.""
# 
