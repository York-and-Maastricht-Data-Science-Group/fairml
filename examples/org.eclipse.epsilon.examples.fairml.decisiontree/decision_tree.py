# load machine learning libraries
import pandas as pd
from sklearn.tree import * 
from sklearn.model_selection import * 
from sklearn import *

# load AI Fair 360 libraries
from aif360.metrics import *
from aif360.algorithms.preprocessing import *
from aif360.datasets.standard_dataset import *
from aif360.explainers import *

from fairml import *

from IPython.display import *

# these variables should be supplied from the parameters
# of the generator
resource = "data/diabetes.csv"
protected_attributes = ['young']
predicted_attribute = 'label'
dropped_attributes = []
test_size = 3.0
training_size = 7.0
validation_size = 0.0
total_size = training_size + validation_size + test_size

# load dataset
data = pd.read_csv(resource, header=0)
# data.head() 

''' Check Original Bias '''
dataset_orig = FairMLDataset(df=data, label_name=predicted_attribute, favorable_classes=[1],
                protected_attribute_names=protected_attributes,
                privileged_classes=[[1]],
                instance_weights_name=None,
                # categorical_features=feature_cols,
                features_to_keep=[],
                features_to_drop=[],
                na_values=[], 
                custom_preprocessing=None,
                metadata=None)
                

dataset_orig_train, dataset_orig_test = dataset_orig.split([training_size/total_size], shuffle=True)
#
privileged_groups = [{protected_attributes[0] : 1}]
unprivileged_groups = [{protected_attributes[0] : 0}]

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
# # display(Markdown("#### Original training dataset"))
explainer_train = MetricTextExplainer(metric_orig_train)
print(explainer_train.disparate_impact())
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

''' Do the training '''
#split dataset in features and target variable
feature_cols = list(data.columns)
feature_cols.remove(predicted_attribute)
for a in dropped_attributes:
    feature_cols.remove(a)
X = data[feature_cols] # Features
y = data[predicted_attribute] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / total_size, random_state=1) 

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
original_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Original Accuracy:", original_accuracy)

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5), dpi=500)
tree.plot_tree(clf,
               feature_names=feature_cols,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('graphics/original_dtree.png')

'''-------------------'''
''' Mitigate the Bias '''
print("Mitigate the Bias ...")
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)


''' Training after mitigation
'''
#split dataset in features and target variable
feature_cols = list(dataset_transf_train.df.columns)
feature_cols.remove(predicted_attribute)
for a in dropped_attributes:
    feature_cols.remove(a)
X = dataset_transf_train.df[feature_cols] # Features
y = dataset_transf_train.df[predicted_attribute] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / total_size, random_state=1)  

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
after_mitigation_accuracy = metrics.accuracy_score(y_test, y_pred)
print("After Mitigation Accuracy:", after_mitigation_accuracy)

plt.figure(figsize=(12, 5), dpi=500)
tree.plot_tree(clf,
               feature_names=feature_cols,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('graphics/mitigated_dtree.png')

''' Check Mitigated Bias '''
metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
explainer_train = MetricTextExplainer(metric_transf_train)
print(explainer_train.disparate_impact())
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())




# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
#
# # Train Decision Tree Classifer
# clf = clf.fit(X_train,y_train)
#
# #Predict the response for test dataset
# y_pred = clf.predict(X_test)
#
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#
# plt.figure(figsize=(8, 3), dpi=200)
# tree.plot_tree(clf,
#                feature_names=feature_cols,
#                class_names=["0", "1"],
#                filled=True,
#                rounded=True);
# plt.savefig('graphics/diabetes.png')