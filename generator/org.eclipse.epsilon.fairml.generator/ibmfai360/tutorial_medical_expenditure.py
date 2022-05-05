#!/usr/bin/env python
# coding: utf-8

# # Medical Expenditure Tutorial

# ## This tutorial demonstrates classification model learning with bias mitigation as a part of a Care Management use case using Medical Expenditure data.

# The notebook demonstrates how the AIF 360 toolkit can be used to detect and reduce bias when learning classifiers using a variety of fairness metrics and algorithms . It also demonstrates how explanations can be generated for predictions made by models learnt with the toolkit using LIME.
# 
# Classifiers are built using Logistic Regression as well as Random Forests.
# 
# Bias detection is demonstrated using several metrics, including disparate impact, average odds difference, statistical parity difference, equal opportunity difference, and Theil index.
# 
# Bias alleviation is explored via a variety of methods, including reweighing (pre-processing algorithm), prejudice remover (in-processing algorithm), and disparate impact remover (pre-processing technique).
# 
# Data from the [Medical Expenditure Panel Survey](https://meps.ahrq.gov/mepsweb/) is used in this tutorial. See [Section 2](#2.-Data-used) below for more details.
# 

# ## Table of Contents

# To return to the table of contents, click on the number at any major section heading.
# 
# [1. Use case](#1.-Use-case)
# 
# [2. Data used](#2.-Data-used)
# 
# [3. Training models without debiasing](#3.-Training-models-on-original-2015-Panel-19-data)
# 
# [4. Reweighing (pre-processing bias mitigation)](#4.-Bias-mitigation-using-pre-processing-technique---Reweighing)
# 
# [5. Prejudice Remover (in-processing bias mitigation)](#5.-Bias-mitigation-using-in-processing-technique---Prejudice-Remover-(PR))
# 
# [6. Summary of results](#6.-Summary-of-Model-Learning-Results)
# 
# [7. Deploying model](#7.-Deploying-model)
# 
# [8. Generating explanations for model predictions using LIME](#8.-Generating-explanations-for-model-predictions-using-LIME)
# 
# [9. Re-deploying Model](#9.-Re-deploying-Model)
# 
# [10. Overall Summary](#10.-SUMMARY)

# ## [1.](#Table-of-Contents) Use case

# In order to demonstrate how AIF 360 can be used to detect and mitigate bias in classfier models, we adopt the following use case:
# 
# 1. a data scientist develops a 'fair' healthcare utilization scoring model with respect to defined protected classes. Fairness may be dictated by legal or government regulations, such as a requirement that additional care decisions be not predicated on factors such as race of the patient.
# 
# 
# 2. developer takes the model AND performance characteristics / specs of the model (e.g. accuracy, fairness tests, etc. basically the model factsheet) and deploys the model in an enterprise app that prioritizes cases for care management.
# 
# 
# 3. the app is put into production and starts scoring people and making recommendations. 
# 
# 
# 4. explanations are generated for each recommendation
# 
# 
# 5. both recommendations and associated explanations are given to nurses as a part of the care management process. The nurses can evaluate the recommendations for quality and correctness and provide feedback.
# 
# 
# 6. nurse feedback as well as analysis of usage data with respect to specs of the model w.r.t accuracy and fairness is communicated to AI Ops specialist and LOB user periodically.
# 
# 
# 7. when significant drift in model specs relative to the model factsheet is observed, the model is sent back for retraining.

# ## [2.](#Table-of-Contents) Data used

# The specific data used is the [2015 Full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) as well as the [2016 Full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192).

# The 2015 file contains data from rounds 3,4,5 of panel 19 (2014) and rounds 1,2,3 of panel 20 (2015). The 2016 file contains data from rounds 3,4,5 of panel 20 (2015) and rounds 1,2,3 of panel 21 (2016).
# 
# For this demonstration, three datasets were constructed: one from panel 19, round 5 (used for learning models), one from panel 20, round 3 (used for deployment/testing of model - steps); the other from panel 21, round 3 (used for re-training and deployment/testing of updated model).

# For each dataset, the sensitive attribute is 'RACE' constructed as follows: 'Whites' (privileged class) defined by the features RACEV2X = 1 (White) and HISPANX = 2 (non Hispanic); 'Non-Whites' that included everyone else.  
# 
# Along with race as the sensitive feature, other features used for modeling include demographics  (such as age, gender, active duty status), physical/mental health assessments, diagnosis codes (such as history of diagnosis of cancer, or diabetes), and limitations (such as cognitive or hearing or vision limitation).
# 
# To measure utilization, a composite feature, 'UTILIZATION', was created to measure the total number of trips requiring some sort of medical care by summing up the following features: OBTOTV15(16), the number of office based visits;  OPTOTV15(16), the number of outpatient visits; ERTOT15(16), the number of ER visits;  IPNGTD15(16), the number of inpatient nights, and  + HHTOTD16, the number of home health visits.
# 
# The model classification task is to predict whether a person would have 'high' utilization (defined as UTILIZATION >= 10, roughly the average utilization for the considered population). High utilization respondents constituted around 17% of each dataset.
# 
# To simulate the scenario, each dataset is split into 3 parts: a train, a validation, and a test/deployment part.
# 
# We assume that the model is initially built and tuned using the 2015 Panel 19 train/test data. (Use case steps 1-2.)
# It is then put into practice and used to score people to identify potential candidates for care management (Use case steps 3-5). Initial deployment is simulated to 2015 Panel 20 deployment data. To show change in performance and/or fairness over time, (use case steps 6-7), the 2016 Panel 21 deployment data is used. Finally, if drift is observed, the 2015 train/validation data is used to learn a new model and evaluated again on the 2016 deployment data

# ## [3.](#Table-of-Contents) Training models on original 2015 Panel 19 data

# First, load all necessary packages

# In[1]:


import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer

np.random.seed(1)


# ### 3.1. Load data & create splits for learning/validating/testing model

# Get the dataset and split into train (50%), validate (30%), and test (20%)

# In[2]:


(dataset_orig_panel19_train,
 dataset_orig_panel19_val,
 dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)

sens_ind = 0
sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]


# This function will be used throughout the notebook to print out some labels, names, etc.

# In[3]:


def describe(train=None, val=None, test=None):
    if train is not None:
        display(Markdown("#### Training Dataset shape"))
        print(train.features.shape)
    if val is not None:
        display(Markdown("#### Validation Dataset shape"))
        print(val.features.shape)
    display(Markdown("#### Test Dataset shape"))
    print(test.features.shape)
    display(Markdown("#### Favorable and unfavorable labels"))
    print(test.favorable_label, test.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(test.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(test.privileged_protected_attributes, 
          test.unprivileged_protected_attributes)
    display(Markdown("#### Dataset feature names"))
    print(test.feature_names)


# Show 2015 dataset details

# In[4]:


describe(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test)


# Metrics for original data

# In[5]:


metric_orig_panel19_train = BinaryLabelDatasetMetric(
        dataset_orig_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)

print(explainer_orig_panel19_train.disparate_impact())


# ### 3.2. Learning a Logistic Regression (LR) classifier on original data

# #### 3.2.1. Training LR model on original data

# In[6]:


dataset = dataset_orig_panel19_train
model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


# #### 3.2.2. Validating LR model on original data

# This function will be used throughout the tutorial to find best threshold using a validation set

# In[7]:


from collections import defaultdict

def test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs


# In[8]:


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val,
                   model=lr_orig_panel19,
                   thresh_arr=thresh_arr)
lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])


# Plot metrics with twin x-axes

# In[9]:


def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)


# Here we plot $1 - \min(\text{disparate impact}, 1/\text{disparate impact})$ since it's possible to overcorrect and end up with a value greater than 1, implying unfairness for the original privileged group. For shorthand, we simply call this 1-min(DI, 1/DI) from now on. We want the plotted metric to be less than 0.2.

# In[10]:


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')


# In[11]:


plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


# Make a function to print out accuracy and fairness metrics. This will be used throughout the tutorial.

# In[12]:


def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))


# In[13]:


describe_metrics(val_metrics, thresh_arr)


# #### 3.2.3. Testing LR model on original data

# In[14]:


lr_orig_metrics = test(dataset=dataset_orig_panel19_test,
                       model=lr_orig_panel19,
                       thresh_arr=[thresh_arr[lr_orig_best_ind]])


# In[15]:


describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])


# For all the fairness metrics displayed above, the value should be close to '0' for fairness.
# 
# 1-min(DI, 1/DI) < 0.2 is typically desired for classifier predictions to be fair.
# 
# However, for a logistic regression classifier trained with original training data, at the best classification rate, this is quite high. This implies unfairness.
# 
# Similarly, $\text{average odds difference} = \frac{(FPR_{unpriv}-FPR_{priv})+(TPR_{unpriv}-TPR_{priv})}{2}$ must be close to zero for the classifier to be fair.
# 
# Again, the results for this classifier-data combination are still high. This still implies unfairness.

# ### 3.3. Learning a Random Forest (RF) classifier on original data

# #### 3.3.1. Training RF model on original data

# In[16]:


dataset = dataset_orig_panel19_train
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
rf_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


# #### 3.3.2. Validating RF model on original data

# In[17]:


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val,
                   model=rf_orig_panel19,
                   thresh_arr=thresh_arr)
rf_orig_best_ind = np.argmax(val_metrics['bal_acc'])


# In[18]:


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')


# In[19]:


plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


# In[20]:


describe_metrics(val_metrics, thresh_arr)


# #### 3.3.3. Testing RF model on original data

# In[21]:


rf_orig_metrics = test(dataset=dataset_orig_panel19_test,
                       model=rf_orig_panel19,
                       thresh_arr=[thresh_arr[rf_orig_best_ind]])


# In[22]:


describe_metrics(rf_orig_metrics, [thresh_arr[rf_orig_best_ind]])


# As in the case of the logistic regression classifier learned on the original data, the fairness metrics for the random forest classifier have values that are quite far from 0.
# 
# For example, 1 - min(DI, 1/DI) has a value of over 0.5 as opposed to the desired value of < 0.2.
# 
# This indicates that the random forest classifier learned on the original data is also unfair.

# ## [4.](#Table-of-Contents) Bias mitigation using pre-processing technique - Reweighing

# ### 4.1. Transform data

# In[23]:


RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)


# Metrics for transformed data

# In[24]:


metric_transf_panel19_train = BinaryLabelDatasetMetric(
        dataset_transf_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_transf_panel19_train = MetricTextExplainer(metric_transf_panel19_train)

print(explainer_transf_panel19_train.disparate_impact())


# ### 4.2. Learning a Logistic Regression (LR) classifier on data transformed by reweighing

# #### 4.2.1. Training LR model after reweighing

# In[25]:


dataset = dataset_transf_panel19_train
model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


# #### 4.2.2. Validating  LR model after reweighing

# In[26]:


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val,
                   model=lr_transf_panel19,
                   thresh_arr=thresh_arr)
lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])


# In[27]:


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')


# In[28]:


plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


# In[29]:


describe_metrics(val_metrics, thresh_arr)


# #### 4.2.3. Testing  LR model after reweighing

# In[30]:


lr_transf_metrics = test(dataset=dataset_orig_panel19_test,
                         model=lr_transf_panel19,
                         thresh_arr=[thresh_arr[lr_transf_best_ind]])


# In[31]:


describe_metrics(lr_transf_metrics, [thresh_arr[lr_transf_best_ind]])


# The fairness metrics for the logistic regression model learned after reweighing are well improved, and thus the model is much more fair relative to the logistic regression model learned from the original data.

# ### 4.3. Learning a Random Forest (RF) classifier on data transformed by reweighing

# #### 4.3.1. Training  RF model after reweighing

# In[32]:


dataset = dataset_transf_panel19_train
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
rf_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


# #### 4.3.2. Validating  RF model after reweighing

# In[33]:


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val,
                   model=rf_transf_panel19,
                   thresh_arr=thresh_arr)
rf_transf_best_ind = np.argmax(val_metrics['bal_acc'])


# In[34]:


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')


# In[35]:


plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


# In[36]:


describe_metrics(val_metrics, thresh_arr)


# #### 4.3.3. Testing  RF model after reweighing

# In[37]:


rf_transf_metrics = test(dataset=dataset_orig_panel19_test,
                         model=rf_transf_panel19,
                         thresh_arr=[thresh_arr[rf_transf_best_ind]])


# In[38]:


describe_metrics(rf_transf_metrics, [thresh_arr[rf_transf_best_ind]])


# Once again, the model learned from the transformed data is fairer than that learned from the original data. However, the random forest model learned from the transformed data is still relatively unfair as compared to the logistic regression model learned from the transformed data.

# ## [5.](#Table-of-Contents) Bias mitigation using in-processing technique - Prejudice Remover (PR)

# ### 5.1. Learning a Prejudice Remover (PR) model on original data

# #### 5.1.1. Training a PR model

# In[39]:


model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
pr_orig_scaler = StandardScaler()

dataset = dataset_orig_panel19_train.copy()
dataset.features = pr_orig_scaler.fit_transform(dataset.features)

pr_orig_panel19 = model.fit(dataset)


# #### 5.1.2. Validating PR model

# In[40]:


thresh_arr = np.linspace(0.01, 0.50, 50)

dataset = dataset_orig_panel19_val.copy()
dataset.features = pr_orig_scaler.transform(dataset.features)

val_metrics = test(dataset=dataset,
                   model=pr_orig_panel19,
                   thresh_arr=thresh_arr)
pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])


# In[41]:


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')


# In[42]:


plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


# In[43]:


describe_metrics(val_metrics, thresh_arr)


# #### 5.1.3. Testing PR model

# In[44]:


dataset = dataset_orig_panel19_test.copy()
dataset.features = pr_orig_scaler.transform(dataset.features)

pr_orig_metrics = test(dataset=dataset,
                       model=pr_orig_panel19,
                       thresh_arr=[thresh_arr[pr_orig_best_ind]])


# In[45]:


describe_metrics(pr_orig_metrics, [thresh_arr[pr_orig_best_ind]])


# As in the case of reweighing, prejudice remover results in a fair model. However, it has come at the expense of relatively lower balanced accuracy.

# ## [6.](#Table-of-Contents) Summary of Model Learning Results

# In[46]:


import pandas as pd
pd.set_option('display.multi_sparse', False)
results = [lr_orig_metrics, rf_orig_metrics, lr_transf_metrics,
           rf_transf_metrics, pr_orig_metrics]
debias = pd.Series(['']*2 + ['Reweighing']*2
                 + ['Prejudice Remover'],
                   name='Bias Mitigator')
clf = pd.Series(['Logistic Regression', 'Random Forest']*2 + [''],
                name='Classifier')
pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index([debias, clf])


# Of all the models, the logistic regression model gives the best balance in terms of balanced accuracy and fairness. While the model learnt by prejudice remover is slightly fairer, it has much lower accuracy. All other models are quite unfair compared to the logistic model. Hence, we take the logistic regression model learnt from data transformed by re-weighing and 'deploy' it.

# ## [7.](#Table-of-Contents) Deploying model

# ### 7.1. Testing model learned on 2014 (Panel 19) on 2015 (Panel 20) deployment data

# In[47]:


dataset_orig_panel20_deploy = MEPSDataset20()

# now align it with the 2014 dataset
dataset_orig_panel20_deploy = dataset_orig_panel19_train.align_datasets(dataset_orig_panel20_deploy)


# In[48]:


# describe(dataset_orig_panel20_train, dataset_orig_panel20_val, dataset_orig_panel20_deploy)
describe(test=dataset_orig_panel20_deploy)


# In[49]:


metric_orig_panel20_deploy = BinaryLabelDatasetMetric(
        dataset_orig_panel20_deploy, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_orig_panel20_deploy = MetricTextExplainer(metric_orig_panel20_deploy)

print(explainer_orig_panel20_deploy.disparate_impact())


# In[50]:


lr_transf_metrics_panel20_deploy = test(
        dataset=dataset_orig_panel20_deploy,
        model=lr_transf_panel19,
        thresh_arr=[thresh_arr[lr_transf_best_ind]])


# In[51]:


describe_metrics(lr_transf_metrics_panel20_deploy, [thresh_arr[lr_transf_best_ind]])


# Deployed model tested on the 2015 Panel 20 data still exhibits fairness as well as maintains accuracy.

# ## [8.](#Table-of-Contents) Generating explanations for model predictions using LIME

# ### 8.1. Generating explanations on 2015 Panel 20 deployment data

# This section shows how LIME can be integrated with AIF360 to get explanations for model predictions.

# In[52]:


train_dataset = dataset_transf_panel19_train  # data the deployed model (lr from transformed data)
test_dataset = dataset_orig_panel20_deploy  # the data model is being tested on
model = lr_transf_panel19  # lr_transf_panel19 is LR model learned from Panel 19 with Reweighing
thresh_arr = np.linspace(0.01, 0.5, 50)
best_thresh = thresh_arr[lr_transf_best_ind]


# First, we need to fit the encoder to the aif360 dataset

# In[53]:


lime_data = LimeEncoder().fit(train_dataset)


# The `transform()` method is then used to convert aif360 features to LIME-compatible features

# In[54]:


s_train = lime_data.transform(train_dataset.features)
s_test = lime_data.transform(test_dataset.features)


# The `LimeTabularExplainer` takes as input the LIME-compatible data along with various other arguments to create a lime explainer

# In[55]:


explainer = LimeTabularExplainer(
        s_train, class_names=lime_data.s_class_names, 
        feature_names=lime_data.s_feature_names,
        categorical_features=lime_data.s_categorical_features, 
        categorical_names=lime_data.s_categorical_names, 
        kernel_width=3, verbose=False, discretize_continuous=True)


# The `inverse_transform()` function is used to transform LIME-compatible data back to aif360-compatible data since that is needed by the model to make predictions. The function below is used to produce the predictions for any perturbed data that is produce by LIME

# In[56]:


def s_predict_fn(x):
    return model.predict_proba(lime_data.inverse_transform(x))


# The `explain_instance()` method can then be used to produce explanations for any instance in the test dataset

# In[57]:


def show_explanation(ind):
    exp = explainer.explain_instance(s_test[ind], s_predict_fn, num_features=10)
    print("Actual label: " + str(test_dataset.labels[ind]))
    exp.as_pyplot_figure()
    plt.show(block=False)


# In[58]:


print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(best_thresh))
show_explanation(0)
show_explanation(2)


# See the [LIME documentation](https://github.com/marcotcr/lime) for detailed description of results. In short, the left hand side shows the label predictions made by the model, the middle shows the features that are important to the instance in question and their contributions (weights) to the label prediction, while the right hand side shows the actual values of the features in the particular instance.

# ## [9.](#Table-of-Contents) Re-deploying Model

# ### 9.1. Testing model learned on 2014 (Panel 19) data on 2016 (Panel 21) deployment data

# Load the Panel 21 data, and split it again into 3 parts: train, validate, and deploy. We test the deployed model against the deployment data. If a new model needs to be learnt, it will be learnt from the train/validate data and then tested again on the deployment data.

# In[59]:


dataset_orig_panel21_deploy = MEPSDataset21()

# now align it with the panel19 datasets
dataset_orig_panel21_deploy = dataset_orig_panel19_train.align_datasets(dataset_orig_panel21_deploy)

describe(test=dataset_orig_panel21_deploy)


# In[60]:


metric_orig_panel21_deploy = BinaryLabelDatasetMetric(
        dataset_orig_panel21_deploy, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_orig_panel21_deploy = MetricTextExplainer(metric_orig_panel21_deploy)

print(explainer_orig_panel21_deploy.disparate_impact())


# Now, the logistic regression classifier trained on the panel 19 data after reweighing is tested against the panel 21 deployment data.

# In[61]:


lr_transf_metrics_panel21_deploy = test(
        dataset=dataset_orig_panel21_deploy,
        model=lr_transf_panel19,
        thresh_arr=[thresh_arr[lr_transf_best_ind]])


# In[62]:


describe_metrics(lr_transf_metrics_panel21_deploy, [thresh_arr[lr_transf_best_ind]])


# Compared to the 2015 panel 20 deployment data results, the $|1 - \text{disparate impact}|$ fairness metric shows a noticable drift upwards. While still within specs, it may be worthwhile to re-learn the model. So even though the model is still relatively fair and accurate, we go ahead and re-learn the model from the 2015 Panel 20 data.

# ### 9.2. Re-learning model (from 2015 Panel 20 data)

# In[63]:


(dataset_orig_panel20_train,
 dataset_orig_panel20_val,
 dataset_orig_panel20_test) = MEPSDataset20().split([0.5, 0.8], shuffle=True) 

# now align them with the 2014 datasets
dataset_orig_panel20_train = dataset_orig_panel19_train.align_datasets(dataset_orig_panel20_train)
dataset_orig_panel20_val = dataset_orig_panel19_train.align_datasets(dataset_orig_panel20_val)
dataset_orig_panel20_test = dataset_orig_panel19_train.align_datasets(dataset_orig_panel20_test)


# **Train and evaluate  new model on 'transformed' 2016 training/test data**

# In[64]:


RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
RW.fit(dataset_orig_panel20_train)
dataset_transf_panel20_train = RW.transform(dataset_orig_panel20_train)


# In[65]:


metric_transf_panel20_train = BinaryLabelDatasetMetric(
        dataset_transf_panel20_train, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_transf_panel20_train = MetricTextExplainer(metric_transf_panel20_train)

print(explainer_transf_panel20_train.disparate_impact())


# In[66]:


dataset = dataset_transf_panel20_train
model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
lr_transf_panel20 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


# In[67]:


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel20_val,
                   model=lr_transf_panel20,
                   thresh_arr=thresh_arr)
lr_transf_best_ind_panel20 = np.argmax(val_metrics['bal_acc'])


# In[68]:


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')


# In[69]:


plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')


# In[70]:


describe_metrics(val_metrics, thresh_arr)


# In[71]:


lr_transf_metrics_panel20_test = test(
        dataset=dataset_orig_panel20_test,
        model=lr_transf_panel20,
        thresh_arr=[thresh_arr[lr_transf_best_ind_panel20]])


# In[72]:


describe_metrics(lr_transf_metrics_panel20_test, [thresh_arr[lr_transf_best_ind_panel20]])


# The new model is both relatively fair as well as accurate so we deploy and test against the 2016 deployment data

# ### 9.3. Testing model learned on 2015 (Panel 20) data on 2016 (Panel 21) deployment data

# **Evaluate new 2015 transformed data model and evaluate again on 2016 deployment data**

# In[73]:


lr_transf_panel20_metrics_panel21_deploy = test(
        dataset=dataset_orig_panel21_deploy,
        model=lr_transf_panel20,
        thresh_arr=[thresh_arr[lr_transf_best_ind_panel20]])


# In[74]:


describe_metrics(lr_transf_panel20_metrics_panel21_deploy, [thresh_arr[lr_transf_best_ind_panel20]])


# The new transformed 2016 data model is again within original accuracy/fairness specs so is deployed

# ## [10.](#Table-of-Contents) SUMMARY

# In[75]:


results = [lr_orig_metrics, lr_transf_metrics,
           lr_transf_metrics_panel20_deploy,
           lr_transf_metrics_panel21_deploy,
           lr_transf_metrics_panel20_test,
           lr_transf_panel20_metrics_panel21_deploy]
debias = pd.Series([''] + ['Reweighing']*5, name='Bias Mitigator')
clf = pd.Series(['Logistic Regression']*6, name='Classifier')
tr = pd.Series(['Panel19']*4 + ['Panel20']*2, name='Training set')
te = pd.Series(['Panel19']*2 + ['Panel20', 'Panel21']*2, name='Testing set')
pd.concat([pd.DataFrame(m) for m in results], axis=0).set_index([debias, clf, tr, te])

