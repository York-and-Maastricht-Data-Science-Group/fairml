#!/usr/bin/env python
# coding: utf-8

# # Detecting and mitigating racial bias in income estimation 
# 
# The goal of this tutorial is to introduce the basic functionality of AI Fairness 360 to an interested developer who may not have a background in bias detection and mitigation.
# 
# *Note: This demo is very similar to the [Credit Scoring Tutorial](tutorial_credit_scoring.ipynb). It is meant as an alternative introduction using a different dataset and mitigation algorithm.*
# 
# ### Biases and Machine Learning
# A machine learning model makes predictions of an outcome for a particular instance. (Given an instance of a loan application, predict if the applicant will repay the loan.) The model makes these predictions based on a training dataset, where many other instances (other loan applications) and actual outcomes (whether they repaid) are provided. Thus, a machine learning algorithm will attempt to find patterns, or generalizations, in the training dataset to use when a prediction for a new instance is needed. (For example, one pattern it might discover is "if a person has salary > USD 40K and has outstanding debt < USD 5, they will repay the loan".) In many domains this technique, called supervised machine learning, has worked very well.
# 
# However, sometimes the patterns that are found may not be desirable or may even be illegal. For example, a loan repay model may determine that age plays a significant role in the prediction of repayment because the training dataset happened to have better repayment for one age group than for another. This raises two problems: 1) the training dataset may not be representative of the true population of people of all age groups, and 2) even if it is representative, it is illegal to base any decision on a applicant's age, regardless of whether this is a good prediction based on historical data.
# 
# AI Fairness 360 is designed to help address this problem with _fairness metrics_ and _bias mitigators_.  Fairness metrics can be used to check for bias in machine learning workflows.  Bias mitigators can be used to overcome bias in the workflow to produce a more fair outcome. 
# 
# The loan scenario describes an intuitive example of illegal bias. However, not all undesirable bias in machine learning is illegal it may also exist in more subtle ways.  For example, a loan company may want a diverse portfolio of customers across all income levels, and thus, will deem it undesirable if they are making more loans to high income levels over low income levels.  Although this is not illegal or unethical, it is undesirable for the company's strategy.
# 
# As these two examples illustrate, a bias detection and/or mitigation toolkit needs to be tailored to the particular bias of interest.  More specifically, it needs to know the attribute or attributes, called _protected attributes_, that are of interest: race is one example of a _protected attribute_ and age is a second.
# 
# ### The Machine Learning Workflow
# To understand how bias can enter a machine learning model, we first review the basics of how a model is created in a supervised machine learning process.  
# 
# 
# 
# ![image](images/Complex_NoProc_V3.jpg)
# 
# 
# 
# 
# 
# 
# 
# 
# First, the process starts with a _training dataset_, which contains a sequence of instances, where each instance has two components: the features and the correct prediction for those features.  Next, a machine learning algorithm is trained on this training dataset to produce a machine learning model.  This generated model can be used to make a prediction when given a new instance.  A second dataset with features and correct predictions, called a _test dataset_, is used to assess the accuracy of the model.
# Since this test dataset is the same format as the training dataset, a set of instances of features and prediction pairs, often these two datasets derive from the same initial dataset.  A random partitioning algorithm is used to split the initial dataset into training and test datasets.
# 
# Bias can enter the system in any of the three steps above.  The training data set may be biased in that its outcomes may be biased towards particular kinds of instances.  The algorithm that creates the model may be biased in that it may generate models that are weighted towards particular features in the input. The test data set may be biased in that it has expectations on correct answers that may be biased.  These three points in the machine learning process represent points for testing and mitigating bias.  In AI Fairness 360 codebase, we call these points _pre-processing_, _in-processing_, and _post-processing_. 
# 
# ### AI Fairness 360
# We are now ready to utilize AI Fairness 360 (`aif360`) to detect and mitigate bias.  We will use the Adult Census Income dataset, splitting it into a training and test dataset.  We will look for bias in the creation of a machine learning model to predict if an individual's annual income exceeds $50,000 based on various personal attributes.  The protected attribute will be "race", with "1" (white) and "0" (not white) being the values for the privileged and unprivileged groups, respectively.
# For this first tutorial, we will check for bias in the initial training data, mitigate the bias, and recheck.  More sophisticated machine learning workflows are given in the author tutorials and demo notebooks in the codebase.
# 
# Here are the steps involved
# #### Step 1: Write import statements
# #### Step 2: Set bias detection options, load dataset, and split between train and test
# #### Step 3: Compute fairness metric on original training dataset
# #### Step 4: Mitigate bias by transforming the original dataset
# #### Step 5: Compute fairness metric on transformed training dataset
# 
# ### Step 1 Import Statements
# As with any Python program, the first step will be to import the necessary packages.  Below we import several components from the aif360 package.  We import a custom version of the AdultDataset with certain features binned, metrics to check for bias, and classes related to the algorithm we will use to mitigate bias. We also import some other non-aif360 useful packages.

# In[1]:


import sys
sys.path.append("../")

import numpy as np

from aif360.metrics import BinaryLabelDatasetMetric

from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions            import load_preproc_data_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions            import get_distortion_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

from IPython.display import Markdown, display


# In[2]:


np.random.seed(1)


# ### Step 2 Load dataset, specifying protected attribute, and split dataset into train and test
# In Step 2 we load the initial dataset, setting the protected attribute to be race.  We then splits the original dataset into training and testing datasets.  Although we will use only  the training dataset in this tutorial, a normal workflow would also use a test dataset for assessing the efficacy (accuracy, fairness, etc.) during the development of a machine learning model.  Finally, we set two variables (to be used in Step 3) for the privileged (1) and unprivileged (0) values for the race attribute.  These are key inputs for detecting and mitigating bias, which will be Step 3 and Step 4.  

# In[3]:


dataset_orig = load_preproc_data_adult(['race'])

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'race': 1}] # White
unprivileged_groups = [{'race': 0}] # Not white


# ### Step 3 Compute fairness metric on original training dataset
# Now that we've identified the protected attribute 'race' and defined privileged and unprivileged values, we can use aif360 to detect bias in the dataset.  One simple test is to compare the percentage of favorable results for the privileged and unprivileged groups, subtracting the former percentage from the latter.   A negative value indicates less favorable outcomes for the unprivileged groups.  This is implemented in the method called mean_difference on the BinaryLabelDatasetMetric class.  The code below performs this check and displays the output:

# In[4]:


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# ### Step 4 Mitigate bias by transforming the original dataset
# The previous step showed that the privileged group was getting 10.5% more positive outcomes in the training dataset.   Since this is not desirable, we are going to try to mitigate this bias in the training dataset.  As stated above, this is called _pre-processing_ mitigation because it happens before the creation of the model.  
# 
# AI Fairness 360 implements several pre-processing mitigation algorithms.  We will choose the Optimized Preprocess algorithm [1], which is implemented in "OptimPreproc" class in the "aif360.algorithms.preprocessing" directory.  This algorithm will transform the dataset to have more equity in positive outcomes on the protected attribute for the privileged and unprivileged groups.
# 
# The algorithm requires some tuning parameters, which are set in the optim_options variable and passed as an argument along with some other parameters, including the 2 variables containg the unprivileged and privileged groups defined in Step 3.
# 
# We then call the fit and transform methods to perform the transformation, producing a newly transformed training dataset (dataset_transf_train).  Finally, we ensure alignment of features between the transformed and the original dataset to enable comparisons.
# 
# [1] Optimized Pre-Processing for Discrimination Prevention, NIPS 2017, Flavio Calmon, Dennis Wei, Bhanukiran Vinzamuri, Karthikeyan Natesan Ramamurthy, and Kush R. Varshney

# In[5]:


optim_options = {
    "distortion_fun": get_distortion_adult,
    "epsilon": 0.05,
    "clist": [0.99, 1.99, 2.99],
    "dlist": [.1, 0.05, 0]
}
    
OP = OptimPreproc(OptTools, optim_options)

OP = OP.fit(dataset_orig_train)
dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)

dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)


# ### Step 5 Compute fairness metric on transformed dataset
# Now that we have a transformed dataset, we can check how effective it was in removing bias by using the same metric we used for the original training dataset in Step 3.  Once again, we use the function mean_difference in the BinaryLabelDatasetMetric class:

# In[6]:


metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


# We see the mitigation step was very effective, the difference in mean outcomes is now -0.051074.  So we went from a 10.5% advantage for the privileged group to a 5.1% advantage for the privileged group &mdash; a reduction in more than half!

# ### Summary
# The purpose of this tutorial is to give a new user to bias detection and mitigation a gentle introduction to some of the functionality of AI Fairness 360.  A more complete use case would take the next step and see how the transformed dataset impacts the accuracy and fairness of a trained model.  This is implemented in the demo notebook in the examples directory of toolkit, called demo_optim_data_preproc.ipynb.  I highly encourage readers to view that notebook as it is  generalization and extension of this simple tutorial.
# 
# There are many metrics one can use to detect the pressence of bias. AI Fairness 360 provides many of them for your use. Since it is not clear which of these metrics to use, we also provide some guidance. Likewise, there are many different bias mitigation algorithms one can employ, many of which are in AI Fairness 360. Other tutorials will demonstrate the use of some of these metrics and mitigations algorithms.
# 
# As mentioned earlier, both fairness metrics and mitigation algorithms can be performed at various stages of the machine learning pipeline.  We recommend checking for bias as often as possible, using as many metrics are relevant for the application domain.  We also recommend incorporating bias detection in an automated continous integration pipeline to ensure bias awareness as a software project evolves.
