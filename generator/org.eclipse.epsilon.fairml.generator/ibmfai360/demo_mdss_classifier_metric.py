#!/usr/bin/env python
# coding: utf-8

# ## Bias scan using Multi-Dimensional Subset Scan (MDSS)
# 
# "Identifying Significant Predictive Bias in Classifiers" https://arxiv.org/abs/1611.08292
# 
# The goal of bias scan is to identify a subgroup(s) that has significantly more predictive bias than would be expected from an unbiased classifier. There are $\prod_{m=1}^{M}\left(2^{|X_{m}|}-1\right)$ unique subgroups from a dataset with $M$ features, with each feature having $|X_{m}|$ discretized values, where a subgroup is any $M$-dimension
# Cartesian set product, between subsets of feature-values from each feature --- excluding the empty set. Bias scan mitigates this computational hurdle by approximately identifing the most statistically biased subgroup in linear time (rather than exponential).
# 
# 
# We define the statistical measure of predictive bias function, $score_{bias}(S)$ as a likelihood ratio score and a function of a given subgroup $S$. The null hypothesis is that the given prediction's odds are correct for all subgroups in
# 
# $\mathcal{D}$: $H_{0}:odds(y_{i})=\frac{\hat{p}_{i}}{1-\hat{p}_{i}}\ \forall i\in\mathcal{D}$.
# 
# The alternative hypothesis assumes some constant multiplicative bias in the odds for some given subgroup $S$:
# 
# 
# $H_{1}:\ odds(y_{i})=q\frac{\hat{p}_{i}}{1-\hat{p}_{i}},\ \text{where}\ q>1\ \forall i\in S\ \mbox{and}\ q=1\ \forall i\notin S.$
# 
# In the classification setting, each observation's likelihood is Bernoulli distributed and assumed independent. This results in the following scoring function for a subgroup $S$
# 
# \begin{align*}
# score_{bias}(S)= & \max_{q}\log\prod_{i\in S}\frac{Bernoulli(\frac{q\hat{p}_{i}}{1-\hat{p}_{i}+q\hat{p}_{i}})}{Bernoulli(\hat{p}_{i})}\\
# = & \max_{q}\log(q)\sum_{i\in S}y_{i}-\sum_{i\in S}\log(1-\hat{p}_{i}+q\hat{p}_{i}).
# \end{align*}
# Our bias scan is thus represented as: $S^{*}=FSS(\mathcal{D},\mathcal{E},F_{score})=MDSS(\mathcal{D},\hat{p},score_{bias})$.
# 
# where $S^{*}$ is the detected most anomalous subgroup, $FSS$ is one of several subset scan algorithms for different problem settings, $\mathcal{D}$ is a dataset with outcomes $Y$ and discretized features $\mathcal{X}$, $\mathcal{E}$ are a set of expectations or 'normal' values for $Y$, and $F_{score}$ is an expectation-based scoring statistic that measures the amount of anomalousness between subgroup observations and their expectations.
# 
# Predictive bias emphasizes comparable predictions for a subgroup and its observations and Bias scan provides a more general method that can detect and characterize such bias, or poor classifier fit, in the larger space of all possible subgroups, without a priori specification.

# In[1]:


import sys
import itertools
sys.path.append("../")


# In[2]:


from aif360.metrics import BinaryLabelDatasetMetric 
from aif360.metrics.mdss_classification_metric import MDSSClassificationMetric
from aif360.metrics.mdss.ScoringFunctions.Bernoulli import Bernoulli


from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier

from IPython.display import Markdown, display
import numpy as np
import pandas as pd


# In[3]:


from aif360.metrics import BinaryLabelDatasetMetric 


# We'll demonstrate scoring a subset and finding the most anomalous subset with bias scan using the compas dataset.
# 
# We can specify subgroups to be scored or scan for the most anomalous subgroup. Bias scan allows us to decide if we aim to identify bias as `higher` than expected probabilities or `lower` than expected probabilities. Depending on the favourable label, the corresponding subgroup may be categorized as priviledged or unprivileged.

# In[4]:


np.random.seed(0)

dataset_orig = load_preproc_data_compas()

female_group = [{'sex': 1}]
male_group = [{'sex': 0}]


# The dataset has the categorical features one-hot encoded so we'll modify the dataset to convert them back 
# to the categorical featues because scanning one-hot encoded features may find subgroups that are not meaningful eg. a subgroup with 2 race values. 

# In[5]:


dataset_orig_df = pd.DataFrame(dataset_orig.features, columns=dataset_orig.feature_names)

age_cat = np.argmax(dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45', 
                                     'age_cat=Greater than 45']].values, axis=1).reshape(-1, 1)
priors_count = np.argmax(dataset_orig_df[['priors_count=0', 'priors_count=1 to 3', 
                                          'priors_count=More than 3']].values, axis=1).reshape(-1, 1)
c_charge_degree = np.argmax(dataset_orig_df[['c_charge_degree=F', 'c_charge_degree=M']].values, axis=1).reshape(-1, 1)

features = np.concatenate((dataset_orig_df[['sex', 'race']].values, age_cat, priors_count,                            c_charge_degree, dataset_orig.labels), axis=1)
feature_names = ['sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree']


# In[6]:


df = pd.DataFrame(features, columns=feature_names + ['two_year_recid'])


# In[7]:


df.head()


# ### training
# We'll create a structured dataset and then train a simple classifier to predict the probability of the outcome

# In[8]:


from aif360.datasets import StandardDataset
dataset = StandardDataset(df, label_name='two_year_recid', favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[[1], [1]],
                 instance_weights_name=None)


# In[9]:


dataset_orig_train, dataset_orig_test = dataset.split([0.7], shuffle=True)


# In[10]:


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


# In[ ]:





# In[11]:


metric_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                             unprivileged_groups=male_group,
                             privileged_groups=female_group)

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_train.mean_difference())
metric_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                             unprivileged_groups=male_group,
                             privileged_groups=female_group)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_test.mean_difference())


# It shows that overall Females in the dataset have a lower observed recidivism them Males.

# If we train a classifier, the model is likely to pick up this bias in the dataset

# In[12]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', C=1.0, penalty='l2')
clf.fit(dataset_orig_train.features, dataset_orig_train.labels.flatten())


# In[13]:


dataset_bias_test_prob = clf.predict_proba(dataset_orig_test.features)[:,1]


# In[14]:


dff = pd.DataFrame(dataset_orig_test.features, columns=dataset_orig_test.feature_names)
dff['observed'] = pd.Series(dataset_orig_test.labels.flatten(), index=dff.index)
dff['probabilities'] = pd.Series(dataset_bias_test_prob, index=dff.index)


# We'll the create another structured dataset as the classified dataset by assigning the predicted probabilities to the scores attribute

# In[15]:


dataset_bias_test = dataset_orig_test.copy()
dataset_bias_test.scores = dataset_bias_test_prob
dataset_bias_test.labels = dataset_orig_test.labels


# ### bias scoring
# 
# We'll create an instance of the MDSS Classification Metric and assess the apriori defined privileged and unprivileged groups; females and males respectively.

# In[16]:


mdss_classified = MDSSClassificationMetric(dataset_orig_test, dataset_bias_test,
                         unprivileged_groups=male_group,
                         privileged_groups=female_group)


# In[17]:


female_privileged_score = mdss_classified.score_groups(privileged=True)
female_privileged_score


# In[18]:


male_unprivileged_score = mdss_classified.score_groups(privileged=False)
male_unprivileged_score


# It appears there is no multiplicative increase in the odds for females thus no bias towards females and the bias score is negligible. Similarly there is no multiplicative decrease in the odds for males. We can alternate our assumptions of priviledge and unprivileged groups to see if there is some bias.

# In[19]:


mdss_classified = MDSSClassificationMetric(dataset_orig_test, dataset_bias_test,
                         unprivileged_groups=female_group,
                         privileged_groups=male_group)


# In[20]:


male_privileged_score = mdss_classified.score_groups(privileged=True)
male_privileged_score


# In[21]:


female_unprivileged_score = mdss_classified.score_groups(privileged=False)
female_unprivileged_score


# It appears there is some multiplicative increase in the odds of recidivism for male and a multiplicative decrease in the odds for females.

# ### bias scan
# We get the bias score for the apriori defined subgroup but assuming we had no prior knowledge 
# about the predictive bias and wanted to find the subgroups with the most bias, we can apply bias scan to identify the priviledged and unpriviledged groups. The privileged argument is not a reference to a group but the direction for which to scan for bias.

# In[22]:


privileged_subset = mdss_classified.bias_scan(penalty=0.5, privileged=True)
unprivileged_subset = mdss_classified.bias_scan(penalty=0.5, privileged=False)


# In[23]:


print(privileged_subset)
print(unprivileged_subset)


# In[25]:


assert privileged_subset[0]
assert unprivileged_subset[0]


# We can observe that the bias score is higher than the score of the prior groups. These subgroups are guaranteed to be the highest scoring subgroup among the exponentially many subgroups.
# 
# For the purposes of this example, the logistic regression model systematically under estimates the recidivism risk of individuals in the `Non-caucasian`, `less than 25`, `Male` subgroup whereas individuals belonging to the `Causasian`, `Female` are assigned a higher risk than is actually observed. We refer to these subgroups as the `detected privileged group` and `detected unprivileged group` respectively.

# We can create another srtuctured dataset using the new groups to compute other dataset metrics.  

# In[26]:


protected_attr_names = set(privileged_subset[0].keys()).union(set(unprivileged_subset[0].keys()))
dataset_orig_test.protected_attribute_names = list(protected_attr_names)
dataset_bias_test.protected_attribute_names = list(protected_attr_names)

protected_attr = np.where(np.isin(dataset_orig_test.feature_names, list(protected_attr_names)))[0]

dataset_orig_test.protected_attributes = dataset_orig_test.features[:, protected_attr]
dataset_bias_test.protected_attributes = dataset_bias_test.features[:, protected_attr]


# In[27]:


display(Markdown("#### Training Dataset shape"))
print(dataset_bias_test.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_bias_test.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_bias_test.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_bias_test.privileged_protected_attributes, 
      dataset_bias_test.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_bias_test.feature_names)


# In[28]:


# converts from dictionary of lists to list of dictionaries
a = list(privileged_subset[0].values())
subset_values = list(itertools.product(*a))

detected_privileged_groups = []
for vals in subset_values:
    detected_privileged_groups.append((dict(zip(privileged_subset[0].keys(), vals))))
    
a = list(unprivileged_subset[0].values())
subset_values = list(itertools.product(*a))

detected_unprivileged_groups = []
for vals in subset_values:
    detected_unprivileged_groups.append((dict(zip(unprivileged_subset[0].keys(), vals))))


# In[29]:


metric_bias_test = BinaryLabelDatasetMetric(dataset_bias_test, 
                                             unprivileged_groups=detected_unprivileged_groups,
                                             privileged_groups=detected_privileged_groups)

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" 
      % metric_bias_test.mean_difference())


# It appears the detected privileged group have a higher risk of recidivism than the unpriviledged group.

# As noted in the paper, predictive bias is different from predictive fairness so there's no the emphasis in the subgroups having comparable predictions between them. 
# We can investigate the difference in what the model predicts vs what we actually observed as well as the multiplicative difference in the odds of the subgroups.

# In[30]:


to_choose = dff[privileged_subset[0].keys()].isin(privileged_subset[0]).all(axis=1)
temp_df = dff.loc[to_choose]


# In[31]:


"Our detected priviledged group has a size of {}, we observe {} as the average risk of recidivism, but our model predicts {}".format(len(temp_df), temp_df['observed'].mean(), temp_df['probabilities'].mean())


# In[32]:


group_obs = temp_df['observed'].mean()
group_prob = temp_df['probabilities'].mean()

odds_mul = (group_obs / (1 - group_obs)) / (group_prob /(1 - group_prob))
"This is a multiplicative increase in the odds by {}".format(odds_mul)


# In[33]:


assert odds_mul > 1


# In[34]:


to_choose = dff[unprivileged_subset[0].keys()].isin(unprivileged_subset[0]).all(axis=1)
temp_df = dff.loc[to_choose]


# In[35]:


"Our detected unpriviledged group has a size of {}, we observe {} as the average risk of recidivism, but our model predicts {}".format(len(temp_df), temp_df['observed'].mean(), temp_df['probabilities'].mean())


# In[36]:


group_obs = temp_df['observed'].mean()
group_prob = temp_df['probabilities'].mean()

odds_mul = (group_obs / (1 - group_obs)) / (group_prob /(1 - group_prob))
"This is a multiplicative decrease in the odds by {}".format(odds_mul)


# In[37]:


assert odds_mul < 1


# In summary, this notebook demonstrates the use of bias scan to identify subgroups with significant predictive bias, as quantified by a likelihood ratio score, using subset scannig. This allows consideration of not just subgroups of a priori interest or small dimensions, but the space of all possible subgroups of features.
# It also presents opportunity for a kind of bias mitigation technique that uses the multiplicative odds in the over-or-under estimated subgroups to adjust for predictive fairness.

# In[ ]:




