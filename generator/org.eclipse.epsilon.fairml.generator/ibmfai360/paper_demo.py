import numpy as np
np.random.seed(0)

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from IPython.display import Markdown, display

dataset_orig = load_preproc_data_adult()
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())


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
X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()
lmod = LogisticRegression(solver='lbfgs')
lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)
X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()
y_pred = lmod.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred)
print("Accuracy: ".format(lr_acc))

dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
dataset_orig_test_pred.labels = y_pred

# positive class index
pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

metric_test = ClassificationMetric(dataset_orig_test,
                                    dataset_orig_test_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)
lr_aod = metric_test.average_odds_difference()
print("Average odds difference: {}".format(lr_aod))

# ### Exponentiated Gradient Reduction
estimator = LogisticRegression(solver='lbfgs')
np.random.seed(0)  # need for reproducibility
exp_grad_red = ExponentiatedGradientReduction(estimator=estimator,
                                              constraints="EqualizedOdds",
                                              drop_prot_attr=False)
exp_grad_red.fit(dataset_orig_train)
exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)


metric_test = ClassificationMetric(dataset_orig_test,
                                   exp_grad_red_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

egr_acc = metric_test.accuracy()
print("Accuracy: {}".format(egr_acc))
egr_aod = metric_test.average_odds_difference()
print("Average odds difference: {}".format(egr_aod))

