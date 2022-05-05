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

from aif360.sklearn.datasets import fetch_compas
from aif360.sklearn.metrics import mdss_bias_scan, mdss_bias_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from IPython.display import Markdown, display
import numpy as np
import pandas as pd


# We'll demonstrate scoring a subset and finding the most anomalous subset with bias scan using the compas dataset.
# 
# We can specify subgroups to be scored or scan for the most anomalous subgroup. Bias scan allows us to decide if we aim to identify bias as `higher` than expected probabilities or `lower` than expected probabilities. Depending on the favourable label, the corresponding subgroup may be categorized as priviledged or unprivileged.

# In[2]:


np.random.seed(0)

#load the data, reindex and change target class to 0/1
X, y = fetch_compas(usecols=['sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree'])

X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

y = pd.Series(y.factorize(sort=True)[0], index=y.index)

# Quantize priors count between 0, 1-3, and >3
def quantize_priors_count(x):
    if x <= 0:
        return '0'
    elif 1 <= x <= 3:
        return '1 to 3'
    else:
        return 'More than 3'
    
X['priors_count'] = pd.Categorical(X['priors_count'].apply(lambda x: quantize_priors_count(x)),  ordered=True, categories=['0', '1 to 3', 'More than 3'])
enc = OrdinalEncoder()

X_vals = enc.fit_transform(X)


# ### training
# We'll split the dataset and then train a simple classifier to predict the probability of the outcome; (0: Survived, 1: Recidivated)

# In[3]:


np.random.seed(0)

(X_train, X_test,
 y_train, y_test) = train_test_split(X_vals, y, train_size=0.7, random_state=1234567)


# In[4]:


clf = LogisticRegression(solver='lbfgs', C=1.0, penalty='l2')
clf.fit(X_train, y_train)


# In[5]:


test_prob = clf.predict_proba(X_test)[:,1]


# In[6]:


dff = pd.DataFrame(X_test, columns=X.columns)
dff['observed'] = pd.Series(y_test.values)
dff['probabilities'] = pd.Series(test_prob)


# In[7]:


dff.head()


# In this example, we assume that the model makes systematic under or over estimatations of the recidivism risk for certain subgroups and our aim is to identify these subgroups

# ### bias scoring
# 
# We'll call the MDSS Classification Metric and score the test set. The privileged argument indicates the direction for which to scan for bias depending on the positive label. In our case since the positive label is 0, `True` corresponds to checking for lower than expected probabilities and `False` corresponds to checking for higher than expected probabilities.

# In[9]:


females = dff[dff['sex'] == 1]
males = dff[dff['sex'] == 0]


# In[10]:


# get the bias score of females assuming they are privileged
print(mdss_bias_score(females['observed'], females['probabilities'], pos_label=0, privileged=True))

# get the bias score of females assuming they are unprivileged
print(mdss_bias_score(females['observed'], females['probabilities'], pos_label=0, privileged=False))


# In[11]:


# get the bias score of males assuming they are privileged
print(mdss_bias_score(males['observed'], males['probabilities'], pos_label=0, privileged=True))

# get the bias score of males assuming they are unprivileged
print(mdss_bias_score(males['observed'], males['probabilities'], pos_label=0, privileged=False))


# If we assume correctly, then our bias score is going to be higher; thus whichever of the assumptions results in a higher bias score has the most evidence of being true. This means females are likley unprivileged whereas males are likely priviledged by our classifier. Note that the default penalty term added is what results in a negative bias score.

# ### bias scan
# We get the bias score for the apriori defined subgroup but assuming we had no prior knowledge 
# about the predictive bias and wanted to find the subgroups with the most bias, we can apply bias scan to identify the priviledged and unpriviledged groups. The privileged argument is not a reference to a group but the direction for which to scan for bias.

# In[29]:


privileged_subset = mdss_bias_scan(dff['observed'], dff['probabilities'], dataset = dff[dff.columns[:-2]],                                    pos_label=0, penalty=0.5, privileged=True)
unprivileged_subset = mdss_bias_scan(dff['observed'], dff['probabilities'], dataset = dff[dff.columns[:-2]],                                      pos_label=0, penalty=0.5, privileged=False)


# In[30]:


print(privileged_subset)
print(unprivileged_subset)


# In[31]:


enc.categories_


# In[32]:


assert privileged_subset[0]
assert unprivileged_subset[0]


# We can observe that the bias score is higher than the score of the prior groups. These subgroups are guaranteed to be the highest scoring subgroup among the exponentially many subgroups.
# 
# For the purposes of this example, the logistic regression model systematically under estimates the recidivism risk of individuals belonging to the `Female` and `Less than 25` group. Whereas individuals belonging to the `Greater than 45` age group are assigned a higher risk than is actually observed. We refer to these subgroups as the `detected privileged group` and `detected unprivileged group` respectively.

# As noted in the paper, predictive bias is different from predictive fairness so there's no the emphasis in the subgroups having comparable predictions between them. 
# We can investigate the difference in what the model predicts vs what we actually observed as well as the multiplicative difference in the odds of the subgroups.

# In[33]:


to_choose = dff[privileged_subset[0].keys()].isin(privileged_subset[0]).all(axis=1)
temp_df = dff.loc[to_choose]


# In[34]:


"Our detected priviledged group has a size of {}, our model predicts {} probability of recidivism but we observe {} as the mean outcome".format(len(temp_df), temp_df['probabilities'].mean(), temp_df['observed'].mean())


# In[35]:


group_obs = temp_df['observed'].mean()
group_prob = temp_df['probabilities'].mean()

odds_mul = (group_obs / (1 - group_obs)) / (group_prob /(1 - group_prob))
"This is a multiplicative increase in the odds by {}".format(odds_mul)


# In[36]:


assert odds_mul > 1


# In[37]:


to_choose = dff[unprivileged_subset[0].keys()].isin(unprivileged_subset[0]).all(axis=1)
temp_df = dff.loc[to_choose]


# In[38]:


"Our detected unpriviledged group has a size of {}, our model predicts {} probability of recidivism but we observe {} as the mean outcome".format(len(temp_df), temp_df['probabilities'].mean(), temp_df['observed'].mean())


# In[39]:


group_obs = temp_df['observed'].mean()
group_prob = temp_df['probabilities'].mean()

odds_mul = (group_obs / (1 - group_obs)) / (group_prob /(1 - group_prob))
"This is a multiplicative decrease in the odds by {}".format(odds_mul)


# In[40]:


assert odds_mul < 1


# In summary this notebook demonstrates the use of bias scan to identify subgroups with significant predictive bias, as quantified by a likelihood ratio score, using subset scannig. This allows consideration of not just subgroups of a priori interest or small dimensions, but the space of all possible subgroups of features.
# It also presents opportunity for a kind of bias mitigation technique that uses the multiplicative odds in the over-or-under estimated subgroups to adjust for predictive fairness.

# In[ ]:




