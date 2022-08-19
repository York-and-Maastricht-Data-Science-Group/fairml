# # FairML: Tutorial Medical Expenditure
# Medical Expenditure Tutorial. This tutorial demonstrates classification model learning 
# with bias mitigation as a part of a Care Management use case 
# using Medical Expenditure data.
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb

# # Contents
# [FairML: Tutorial Medical Expenditure](#FairML:-Tutorial-Medical-Expenditure)
# [Contents](#Contents)
# * [1. Bias Mitigation: Medical Expenditure Bias Mitigation](#1.-Bias-Mitigation:-Medical-Expenditure-Bias-Mitigation)
#   * [1.1. Dataset Panel19-Panel19](#1.1.-Dataset-Panel19-Panel19)
#       * [1.1.1. Original Dataset](#1.1.1.-Original-Dataset)
#           * [1.1.1.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#1.1.1.1.-Original-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [1.1.1.1.1. Bias Metrics](#1.1.1.1.1.-Original-Dataset:-Bias-Metrics)
#           * [1.1.1.2. Classifier RandomForestClassifier, Parameters: n_estimators=500, min_samples_leaf=25](#1.1.1.2.-Original-Dataset:-Classifier-RandomForestClassifier,-Parameters:-n_estimators=500,-min_samples_leaf=25)
#               * [1.1.1.2.1. Bias Metrics](#1.1.1.2.1.-Original-Dataset:-Bias-Metrics)
#       * [1.1.2. Mitigate Bias using Reweighing](#1.1.2.-Mitigate-Bias-using-Reweighing)
#           * [1.1.2.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#1.1.2.1.-After-mitigation-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [1.1.2.1.1. Bias Metrics](#1.1.2.1.1.-After-mitigation:-Bias-Metrics)
#           * [1.1.2.2. Classifier RandomForestClassifier, Parameters: n_estimators=500, min_samples_leaf=25](#1.1.2.2.-After-mitigation-Dataset:-Classifier-RandomForestClassifier,-Parameters:-n_estimators=500,-min_samples_leaf=25)
#               * [1.1.2.2.1. Bias Metrics](#1.1.2.2.1.-After-mitigation:-Bias-Metrics)
#       * [1.1.3. Mitigate Bias using PrejudiceRemover](#1.1.3.-Mitigate-Bias-using-PrejudiceRemover)
#           * [1.1.3.1. Classifier PrejudiceRemover, Parameters: sensitive_attr='RACE', eta=25.0](#1.1.3.1.-After-mitigation-Dataset:-Classifier-PrejudiceRemover,-Parameters:-sensitive_attr='RACE',-eta=25.0)
#               * [1.1.3.1.1. Bias Metrics](#1.1.3.1.1.-After-mitigation:-Bias-Metrics)
#   * [1.2. Summary](#1.2.-Summary)
# * [2. Bias Mitigation: Model Deployment](#2.-Bias-Mitigation:-Model-Deployment)
#   * [2.1. Dataset Panel19-Panel19](#2.1.-Dataset-Panel19-Panel19)
#       * [2.1.1. Original Dataset](#2.1.1.-Original-Dataset)
#           * [2.1.1.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.1.1.1.-Original-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.1.1.1.1. Bias Metrics](#2.1.1.1.1.-Original-Dataset:-Bias-Metrics)
#       * [2.1.2. Mitigate Bias using Reweighing](#2.1.2.-Mitigate-Bias-using-Reweighing)
#           * [2.1.2.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.1.2.1.-After-mitigation-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.1.2.1.1. Bias Metrics](#2.1.2.1.1.-After-mitigation:-Bias-Metrics)
#   * [2.2. Dataset Panel19-Panel20](#2.2.-Dataset-Panel19-Panel20)
#       * [2.2.1. Original Dataset](#2.2.1.-Original-Dataset)
#           * [2.2.1.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.2.1.1.-Original-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.2.1.1.1. Bias Metrics](#2.2.1.1.1.-Original-Dataset:-Bias-Metrics)
#       * [2.2.2. Mitigate Bias using Reweighing](#2.2.2.-Mitigate-Bias-using-Reweighing)
#           * [2.2.2.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.2.2.1.-After-mitigation-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.2.2.1.1. Bias Metrics](#2.2.2.1.1.-After-mitigation:-Bias-Metrics)
#   * [2.3. Dataset Panel19-Panel21](#2.3.-Dataset-Panel19-Panel21)
#       * [2.3.1. Original Dataset](#2.3.1.-Original-Dataset)
#           * [2.3.1.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.3.1.1.-Original-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.3.1.1.1. Bias Metrics](#2.3.1.1.1.-Original-Dataset:-Bias-Metrics)
#       * [2.3.2. Mitigate Bias using Reweighing](#2.3.2.-Mitigate-Bias-using-Reweighing)
#           * [2.3.2.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.3.2.1.-After-mitigation-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.3.2.1.1. Bias Metrics](#2.3.2.1.1.-After-mitigation:-Bias-Metrics)
#   * [2.4. Dataset Panel20-Panel20](#2.4.-Dataset-Panel20-Panel20)
#       * [2.4.1. Original Dataset](#2.4.1.-Original-Dataset)
#           * [2.4.1.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.4.1.1.-Original-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.4.1.1.1. Bias Metrics](#2.4.1.1.1.-Original-Dataset:-Bias-Metrics)
#       * [2.4.2. Mitigate Bias using Reweighing](#2.4.2.-Mitigate-Bias-using-Reweighing)
#           * [2.4.2.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.4.2.1.-After-mitigation-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.4.2.1.1. Bias Metrics](#2.4.2.1.1.-After-mitigation:-Bias-Metrics)
#   * [2.5. Dataset Panel20-Panel21](#2.5.-Dataset-Panel20-Panel21)
#       * [2.5.1. Original Dataset](#2.5.1.-Original-Dataset)
#           * [2.5.1.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.5.1.1.-Original-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.5.1.1.1. Bias Metrics](#2.5.1.1.1.-Original-Dataset:-Bias-Metrics)
#       * [2.5.2. Mitigate Bias using Reweighing](#2.5.2.-Mitigate-Bias-using-Reweighing)
#           * [2.5.2.1. Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1](#2.5.2.1.-After-mitigation-Dataset:-Classifier-LogisticRegression,-Parameters:-solver='liblinear',-random_state=1)
#               * [2.5.2.1.1. Bias Metrics](#2.5.2.1.1.-After-mitigation:-Bias-Metrics)
#   * [2.6. Summary](#2.6.-Summary)

# Load dependencies.

import inspect
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib
from fairml import *
from sklearn import tree
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.kernel_ridge import *
from sklearn.ensemble import *
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from aif360.algorithms.preprocessing import *
from aif360.algorithms.inprocessing import *
from aif360.algorithms.postprocessing import *
from aif360.datasets import StandardDataset
from collections import defaultdict
from IPython.display import Markdown, display
from IPython import get_ipython
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

fairml = FairML()
def get_fairml():
    return fairml

print("========================")
print("FairML: Tutorial Medical Expenditure")
print("========================")
print("Description:")
print("Medical Expenditure Tutorial. This tutorial demonstrates classification model learning "+
"with bias mitigation as a part of a Care Management use case "+
"using Medical Expenditure data."+
"https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb")

# ## [1.](#Contents) Bias Mitigation: Medical Expenditure Bias Mitigation
np.random.seed(0)
bm = fairml.add_bias_mitigation(BiasMitigation())
bm.name = "Medical Expenditure Bias Mitigation"

print("")
print("========================")
print("Bias Mitigation: Medical Expenditure Bias Mitigation")
print("------------------------")

# ### [1.1.](#Contents) Dataset Panel19-Panel19
print("")
print("Dataset: Panel19-Panel19")
print("-------------")

# #### [1.1.1.](#Contents) Original Dataset
bm.predicted_attribute = ''
bm.protected_attributes = ['RACE']
bm.features_to_keep = []
bm.favorable_class = 1
bm.privileged_class = 1
bm.unprivileged_class = 0
bm.dropped_attributes = []
bm.na_values = []
bm.training_size = 5.0    
bm.test_size = 2.0
bm.validation_size = 3.0
bm.total_size = bm.training_size + bm.test_size + bm.validation_size
bm.categorical_features = []
bm.default_mappings = None

# Load dataset
bm.resource = "" 
bm.dataset_original = MEPSDataset19()

bm.dataset_original_train, bm.dataset_original_valid, bm.dataset_original_test = bm.dataset_original.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)




bm.privileged_groups = [{bm.protected_attributes[0] : bm.privileged_class}]
bm.unprivileged_groups = [{bm.protected_attributes[0] : bm.unprivileged_class}]

bm.init_new_result("Original", "", "Panel19-Panel19", "", "")
bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)


# ##### [1.1.1.1.](#Contents) Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")

# Train the model from the original train data
classifier = LogisticRegression(solver='liblinear', random_state=1)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [1.1.1.1.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel19-Panel19", "LogisticRegression", "solver='liblinear', random_state=1")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )

# ##### [1.1.1.2.](#Contents) Original Dataset: Classifier RandomForestClassifier, Parameters: n_estimators=500, min_samples_leaf=25
print("")
print("Original Dataset: Classifier RandomForestClassifier, Parameters: n_estimators=500, min_samples_leaf=25")
print("-------------")

# Train the model from the original train data
classifier = RandomForestClassifier(n_estimators=500, min_samples_leaf=25)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [1.1.1.2.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel19-Panel19", "RandomForestClassifier", "n_estimators=500, min_samples_leaf=25")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [1.1.2.](#Contents) Mitigate Bias using Reweighing  
print("")
print("Mitigate Bias using Reweighing")
print("-------------")
mitigation_method = bm.create_mitigation_method(Reweighing, )
mitigation_method = mitigation_method.fit(bm.dataset_original_train, )
dataset_mitigated_train = mitigation_method.transform(bm.dataset_original_train)
dataset_mitigated_valid = mitigation_method.transform(bm.dataset_original_valid)
dataset_mitigated_test = bm.dataset_original_test



# ##### [1.1.2.1.](#Contents) After-mitigation Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("After-mitigation Training: LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")


# Train the model from the after-mitigation train data

classifier = LogisticRegression(solver='liblinear', random_state=1)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [1.1.2.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel19-Panel19", "LogisticRegression", "solver='liblinear', random_state=1")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# ##### [1.1.2.2.](#Contents) After-mitigation Dataset: Classifier RandomForestClassifier, Parameters: n_estimators=500, min_samples_leaf=25
print("")
print("After-mitigation Training: RandomForestClassifier, Parameters: n_estimators=500, min_samples_leaf=25")
print("-------------")


# Train the model from the after-mitigation train data

classifier = RandomForestClassifier(n_estimators=500, min_samples_leaf=25)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [1.1.2.2.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel19-Panel19", "RandomForestClassifier", "n_estimators=500, min_samples_leaf=25")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [1.1.3.](#Contents) Mitigate Bias using PrejudiceRemover  
print("")
print("Mitigate Bias using PrejudiceRemover")
print("-------------")
dataset_mitigated_train = bm.dataset_original_train.copy(deepcopy=True)
dataset_mitigated_valid = bm.dataset_original_valid.copy(deepcopy=True)
dataset_mitigated_test = bm.dataset_original_test.copy(deepcopy=True)
min_max_scaler = MaxAbsScaler()
dataset_mitigated_train.features = min_max_scaler.fit_transform(bm.dataset_original_train.features)
dataset_mitigated_valid.features = min_max_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_mitigated_test.features = min_max_scaler.fit_transform(bm.dataset_original_test.features)



# ##### [1.1.3.1.](#Contents) After-mitigation Dataset: Classifier PrejudiceRemover, Parameters: sensitive_attr='RACE', eta=25.0
print("")
print("After-mitigation Training: PrejudiceRemover, Parameters: sensitive_attr='RACE', eta=25.0")
print("-------------")


# Train the model from the after-mitigation train data

classifier = bm.create_mitigation_method(PrejudiceRemover, sensitive_attr='RACE', eta=25.0)
model_mitigated_train = classifier.fit(dataset_mitigated_train, )

# ###### [1.1.3.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = model_mitigated_train.predict(dataset_mitigated_train, )
dataset_mitigated_valid_pred = model_mitigated_train.predict(dataset_mitigated_valid, )
dataset_mitigated_test_pred = model_mitigated_train.predict(dataset_mitigated_test, )


bm.init_new_result("PrejudiceRemover", "sensitive_attr='RACE', eta=25.0", "Panel19-Panel19", "PrejudiceRemover", "sensitive_attr='RACE', eta=25.0")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )



# ### [1.2.](#Contents) Summary

table = bm.display_summary()
table.style.set_caption("Worst = White, Best = Green, Mid = Yellow") \
    .apply(bm.highlight_fairest_values, axis=1)

if get_ipython() is not None:
    get_ipython().magic("%matplotlib inline")

bm.display_barchart()

# ## [2.](#Contents) Bias Mitigation: Model Deployment
np.random.seed(0)
bm = fairml.add_bias_mitigation(BiasMitigation())
bm.name = "Model Deployment"

print("")
print("========================")
print("Bias Mitigation: Model Deployment")
print("------------------------")

# ### [2.1.](#Contents) Dataset Panel19-Panel19
print("")
print("Dataset: Panel19-Panel19")
print("-------------")

# #### [2.1.1.](#Contents) Original Dataset
bm.predicted_attribute = ''
bm.protected_attributes = ['RACE']
bm.features_to_keep = []
bm.favorable_class = 1
bm.privileged_class = 1
bm.unprivileged_class = 0
bm.dropped_attributes = []
bm.na_values = []
bm.training_size = 5.0    
bm.test_size = 2.0
bm.validation_size = 3.0
bm.total_size = bm.training_size + bm.test_size + bm.validation_size
bm.categorical_features = []
bm.default_mappings = None

# Load dataset
bm.resource = "" 
bm.dataset_original = MEPSDataset19()

bm.dataset_original_train, bm.dataset_original_valid, bm.dataset_original_test = bm.dataset_original.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)




bm.privileged_groups = [{bm.protected_attributes[0] : bm.privileged_class}]
bm.unprivileged_groups = [{bm.protected_attributes[0] : bm.unprivileged_class}]

bm.init_new_result("Original", "", "Panel19-Panel19", "", "")
bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)


# ##### [2.1.1.1.](#Contents) Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")

# Train the model from the original train data
classifier = LogisticRegression(solver='liblinear', random_state=1)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [2.1.1.1.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel19-Panel19", "LogisticRegression", "solver='liblinear', random_state=1")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [2.1.2.](#Contents) Mitigate Bias using Reweighing  
print("")
print("Mitigate Bias using Reweighing")
print("-------------")
mitigation_method = bm.create_mitigation_method(Reweighing, )
mitigation_method = mitigation_method.fit(bm.dataset_original_train, )
dataset_mitigated_train = mitigation_method.transform(bm.dataset_original_train)
dataset_mitigated_valid = mitigation_method.transform(bm.dataset_original_valid)
dataset_mitigated_test = bm.dataset_original_test



# ##### [2.1.2.1.](#Contents) After-mitigation Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("After-mitigation Training: LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")


# Train the model from the after-mitigation train data

classifier = LogisticRegression(solver='liblinear', random_state=1)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [2.1.2.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel19-Panel19", "LogisticRegression", "solver='liblinear', random_state=1")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# ### [2.2.](#Contents) Dataset Panel19-Panel20
print("")
print("Dataset: Panel19-Panel20")
print("-------------")

# #### [2.2.1.](#Contents) Original Dataset
bm.predicted_attribute = ''
bm.protected_attributes = ['RACE']
bm.features_to_keep = []
bm.favorable_class = 1
bm.privileged_class = 1
bm.unprivileged_class = 0
bm.dropped_attributes = []
bm.na_values = []
bm.training_size = 5.0    
bm.test_size = 2.0
bm.validation_size = 3.0
bm.total_size = bm.training_size + bm.test_size + bm.validation_size
bm.categorical_features = []
bm.default_mappings = None

# Load dataset
bm.resource = "" 
bm.dataset_original = MEPSDataset19()

bm.dataset_original_train, bm.dataset_original_valid, bm.dataset_original_test = bm.dataset_original.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)


bm.test_dataset_module = MEPSDataset20()
bm.test_dataset_module = bm.dataset_original_train.align_datasets(bm.test_dataset_module)
_ , _ , bm.dataset_original_test = bm.test_dataset_module.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)


bm.privileged_groups = [{bm.protected_attributes[0] : bm.privileged_class}]
bm.unprivileged_groups = [{bm.protected_attributes[0] : bm.unprivileged_class}]

bm.init_new_result("Original", "", "Panel19-Panel20", "", "")
bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)


# ##### [2.2.1.1.](#Contents) Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")

# Train the model from the original train data
classifier = LogisticRegression(solver='liblinear', random_state=1)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [2.2.1.1.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel19-Panel20", "LogisticRegression", "solver='liblinear', random_state=1")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [2.2.2.](#Contents) Mitigate Bias using Reweighing  
print("")
print("Mitigate Bias using Reweighing")
print("-------------")
mitigation_method = bm.create_mitigation_method(Reweighing, )
mitigation_method = mitigation_method.fit(bm.dataset_original_train, )
dataset_mitigated_train = mitigation_method.transform(bm.dataset_original_train)
dataset_mitigated_valid = mitigation_method.transform(bm.dataset_original_valid)
dataset_mitigated_test = bm.dataset_original_test



# ##### [2.2.2.1.](#Contents) After-mitigation Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("After-mitigation Training: LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")


# Train the model from the after-mitigation train data

classifier = LogisticRegression(solver='liblinear', random_state=1)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [2.2.2.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel19-Panel20", "LogisticRegression", "solver='liblinear', random_state=1")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# ### [2.3.](#Contents) Dataset Panel19-Panel21
print("")
print("Dataset: Panel19-Panel21")
print("-------------")

# #### [2.3.1.](#Contents) Original Dataset
bm.predicted_attribute = ''
bm.protected_attributes = ['RACE']
bm.features_to_keep = []
bm.favorable_class = 1
bm.privileged_class = 1
bm.unprivileged_class = 0
bm.dropped_attributes = []
bm.na_values = []
bm.training_size = 5.0    
bm.test_size = 2.0
bm.validation_size = 3.0
bm.total_size = bm.training_size + bm.test_size + bm.validation_size
bm.categorical_features = []
bm.default_mappings = None

# Load dataset
bm.resource = "" 
bm.dataset_original = MEPSDataset19()

bm.dataset_original_train, bm.dataset_original_valid, bm.dataset_original_test = bm.dataset_original.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)


bm.test_dataset_module = MEPSDataset21()
bm.test_dataset_module = bm.dataset_original_train.align_datasets(bm.test_dataset_module)
_ , _ , bm.dataset_original_test = bm.test_dataset_module.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)


bm.privileged_groups = [{bm.protected_attributes[0] : bm.privileged_class}]
bm.unprivileged_groups = [{bm.protected_attributes[0] : bm.unprivileged_class}]

bm.init_new_result("Original", "", "Panel19-Panel21", "", "")
bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)


# ##### [2.3.1.1.](#Contents) Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")

# Train the model from the original train data
classifier = LogisticRegression(solver='liblinear', random_state=1)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [2.3.1.1.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel19-Panel21", "LogisticRegression", "solver='liblinear', random_state=1")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [2.3.2.](#Contents) Mitigate Bias using Reweighing  
print("")
print("Mitigate Bias using Reweighing")
print("-------------")
mitigation_method = bm.create_mitigation_method(Reweighing, )
mitigation_method = mitigation_method.fit(bm.dataset_original_train, )
dataset_mitigated_train = mitigation_method.transform(bm.dataset_original_train)
dataset_mitigated_valid = mitigation_method.transform(bm.dataset_original_valid)
dataset_mitigated_test = bm.dataset_original_test



# ##### [2.3.2.1.](#Contents) After-mitigation Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("After-mitigation Training: LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")


# Train the model from the after-mitigation train data

classifier = LogisticRegression(solver='liblinear', random_state=1)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [2.3.2.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel19-Panel21", "LogisticRegression", "solver='liblinear', random_state=1")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# ### [2.4.](#Contents) Dataset Panel20-Panel20
print("")
print("Dataset: Panel20-Panel20")
print("-------------")

# #### [2.4.1.](#Contents) Original Dataset
bm.predicted_attribute = ''
bm.protected_attributes = ['RACE']
bm.features_to_keep = []
bm.favorable_class = 1
bm.privileged_class = 1
bm.unprivileged_class = 0
bm.dropped_attributes = []
bm.na_values = []
bm.training_size = 5.0    
bm.test_size = 2.0
bm.validation_size = 3.0
bm.total_size = bm.training_size + bm.test_size + bm.validation_size
bm.categorical_features = []
bm.default_mappings = None

# Load dataset
bm.resource = "" 
bm.dataset_original = MEPSDataset20()

bm.dataset_original_train, bm.dataset_original_valid, bm.dataset_original_test = bm.dataset_original.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)




bm.privileged_groups = [{bm.protected_attributes[0] : bm.privileged_class}]
bm.unprivileged_groups = [{bm.protected_attributes[0] : bm.unprivileged_class}]

bm.init_new_result("Original", "", "Panel20-Panel20", "", "")
bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)


# ##### [2.4.1.1.](#Contents) Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")

# Train the model from the original train data
classifier = LogisticRegression(solver='liblinear', random_state=1)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [2.4.1.1.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel20-Panel20", "LogisticRegression", "solver='liblinear', random_state=1")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [2.4.2.](#Contents) Mitigate Bias using Reweighing  
print("")
print("Mitigate Bias using Reweighing")
print("-------------")
mitigation_method = bm.create_mitigation_method(Reweighing, )
mitigation_method = mitigation_method.fit(bm.dataset_original_train, )
dataset_mitigated_train = mitigation_method.transform(bm.dataset_original_train)
dataset_mitigated_valid = mitigation_method.transform(bm.dataset_original_valid)
dataset_mitigated_test = bm.dataset_original_test



# ##### [2.4.2.1.](#Contents) After-mitigation Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("After-mitigation Training: LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")


# Train the model from the after-mitigation train data

classifier = LogisticRegression(solver='liblinear', random_state=1)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [2.4.2.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel20-Panel20", "LogisticRegression", "solver='liblinear', random_state=1")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# ### [2.5.](#Contents) Dataset Panel20-Panel21
print("")
print("Dataset: Panel20-Panel21")
print("-------------")

# #### [2.5.1.](#Contents) Original Dataset
bm.predicted_attribute = ''
bm.protected_attributes = ['RACE']
bm.features_to_keep = []
bm.favorable_class = 1
bm.privileged_class = 1
bm.unprivileged_class = 0
bm.dropped_attributes = []
bm.na_values = []
bm.training_size = 5.0    
bm.test_size = 2.0
bm.validation_size = 3.0
bm.total_size = bm.training_size + bm.test_size + bm.validation_size
bm.categorical_features = []
bm.default_mappings = None

# Load dataset
bm.resource = "" 
bm.dataset_original = MEPSDataset20()

bm.dataset_original_train, bm.dataset_original_valid, bm.dataset_original_test = bm.dataset_original.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)


bm.test_dataset_module = MEPSDataset21()
bm.test_dataset_module = bm.dataset_original_train.align_datasets(bm.test_dataset_module)
_ , _ , bm.dataset_original_test = bm.test_dataset_module.split([bm.training_size/bm.total_size, (bm.training_size + bm.validation_size)/bm.total_size], shuffle=True)


bm.privileged_groups = [{bm.protected_attributes[0] : bm.privileged_class}]
bm.unprivileged_groups = [{bm.protected_attributes[0] : bm.unprivileged_class}]

bm.init_new_result("Original", "", "Panel20-Panel21", "", "")
bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups)


# ##### [2.5.1.1.](#Contents) Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("Original Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")

# Train the model from the original train data
classifier = LogisticRegression(solver='liblinear', random_state=1)
model_original_train = bm.train(bm.dataset_original_train,  classifier, without_weight=False)

# ###### [2.5.1.1.1.](#Contents) Original Dataset: Bias Metrics
print("Original Bias Metrics")

dataset_original_train_pred = bm.dataset_original_train.copy(deepcopy=True)
dataset_original_valid_pred = bm.dataset_original_valid.copy(deepcopy=True)
dataset_original_test_pred = bm.dataset_original_test.copy(deepcopy=True)

standard_scaler = StandardScaler()
dataset_original_train_pred_features = standard_scaler.fit_transform(bm.dataset_original_train.features)
dataset_original_valid_pred_features = standard_scaler.fit_transform(bm.dataset_original_valid.features)
dataset_original_test_pred_features = standard_scaler.fit_transform(bm.dataset_original_test.features)

dataset_original_train_pred = bm.create_predicted_dataset(dataset_original_train_pred, model_original_train)
dataset_original_valid_pred = bm.create_predicted_dataset(dataset_original_valid_pred, model_original_train)
dataset_original_test_pred = bm.create_predicted_dataset(dataset_original_test_pred, model_original_train)


pos_ind = np.where(model_original_train.classes_ == bm.dataset_original_train.favorable_label)[0][0]
dataset_original_train_pred.scores = model_original_train.predict_proba(dataset_original_train_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_valid_pred.scores = model_original_train.predict_proba(dataset_original_valid_pred_features)[:,pos_ind].reshape(-1,1)
dataset_original_test_pred.scores = model_original_train.predict_proba(dataset_original_test_pred_features)[:,pos_ind].reshape(-1,1)

bm.init_new_result("Original", "", "Panel20-Panel21", "LogisticRegression", "solver='liblinear', random_state=1")

bm.measure_bias("balanced_accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("mean_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("theil_index", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_original_train,  )
bm.measure_bias("accuracy", baseline_dataset=bm.dataset_original_test, predicted_dataset=dataset_original_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )


# #### [2.5.2.](#Contents) Mitigate Bias using Reweighing  
print("")
print("Mitigate Bias using Reweighing")
print("-------------")
mitigation_method = bm.create_mitigation_method(Reweighing, )
mitigation_method = mitigation_method.fit(bm.dataset_original_train, )
dataset_mitigated_train = mitigation_method.transform(bm.dataset_original_train)
dataset_mitigated_valid = mitigation_method.transform(bm.dataset_original_valid)
dataset_mitigated_test = bm.dataset_original_test



# ##### [2.5.2.1.](#Contents) After-mitigation Dataset: Classifier LogisticRegression, Parameters: solver='liblinear', random_state=1
print("")
print("After-mitigation Training: LogisticRegression, Parameters: solver='liblinear', random_state=1")
print("-------------")


# Train the model from the after-mitigation train data

classifier = LogisticRegression(solver='liblinear', random_state=1)
model_mitigated_train = bm.train(dataset_mitigated_train,  classifier, without_weight=False)

# ###### [2.5.2.1.1.](#Contents) After-mitigation: Bias Metrics

print("After-mitigation Metrics")
dataset_mitigated_train_pred = bm.create_predicted_dataset(dataset_mitigated_train, model_mitigated_train)
dataset_mitigated_valid_pred = bm.create_predicted_dataset(dataset_mitigated_valid, model_mitigated_train)
dataset_mitigated_test_pred = bm.create_predicted_dataset(dataset_mitigated_test, model_mitigated_train)


bm.init_new_result("Reweighing", "", "Panel20-Panel21", "LogisticRegression", "solver='liblinear', random_state=1")


bm.measure_bias("balanced_accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("mean_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("average_odds_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("disparate_impact", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train, plot_threshold=True, )
bm.measure_bias("statistical_parity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("equal_opportunity_difference", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("theil_index", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups, optimal_threshold=True, model=model_mitigated_train,  )
bm.measure_bias("accuracy", baseline_dataset=dataset_mitigated_test, predicted_dataset=dataset_mitigated_test_pred, privileged_groups=bm.privileged_groups, unprivileged_groups=bm.unprivileged_groups,   )



# ### [2.6.](#Contents) Summary

table = bm.display_summary()
table.style.set_caption("Worst = White, Best = Green, Mid = Yellow") \
    .apply(bm.highlight_fairest_values, axis=1)

if get_ipython() is not None:
    get_ipython().magic("%matplotlib inline")

bm.display_barchart()

