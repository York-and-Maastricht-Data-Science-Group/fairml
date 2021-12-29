# load machine learning libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import * 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# load AI Fair 360 libraries
from aif360.metrics import *
from aif360.algorithms.preprocessing import *
from aif360.datasets.standard_dataset import *
from aif360.explainers import *

# load FairML libraries
from fairml import *

# load IPython libraries
from IPython.display import *

print("========================")
print("FairML: Decision Tree")
print("========================")
print("Description:")
print("Predict whether income exceeds $50K/yr based on census data. Also known as \"Census Income\" dataset. <=50K = 0, >50K = 1 female = 0, male = 1 other = 0, white = 1")

print("")
print("========================")
print("Bias Mitigation: Sex Bias Checking with Reweighing")
print("------------------------")
resource = "data/adult.data.csv"
protected_attributes = ['sex']
predicted_attribute = 'class'
dropped_attributes = ['race']
training_size = 7
validation_size = 0
test_size = 3
total_size = training_size + validation_size + test_size
priviledgedGroup = 1 
unprivilegedGroup = 0

# load dataset
data = pd.read_csv(resource, header=0)

dataset_original = FairMLDataset(df=data, label_name=predicted_attribute, favorable_classes=[priviledgedGroup],
                protected_attribute_names=protected_attributes,
                privileged_classes=[[priviledgedGroup]],
                instance_weights_name=None,
                # categorical_features=feature_cols,
                features_to_keep=[],
                features_to_drop=dropped_attributes,
                na_values=[], 
                custom_preprocessing=None,
                metadata=None)
                
dataset_original_train, dataset_original_test = dataset_original.split([training_size/total_size], shuffle=True)
#
privileged_groups = [{protected_attributes[0] : priviledgedGroup}]
unprivileged_groups = [{protected_attributes[0] : unprivilegedGroup}]

print("")
print("Original Bias Checking: Sex Bias Checking")
print("-------------")
print("Dataset: Adult Census Income")
metric_original_train = BinaryLabelDatasetMetric(dataset_original_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
explainer_train = MetricTextExplainer(metric_original_train)
			
print("Original explainer: " + explainer_train.disparate_impact())
print("Original statistical_parity_difference: %f" % metric_original_train.statistical_parity_difference())
print("Original mean_difference: %f" % metric_original_train.mean_difference())


#split dataset in features and target variable
attributes = list(data.columns)
attributes.remove(predicted_attribute)
for a in dropped_attributes:
	attributes.remove(a)
X = data[attributes] # Features
y = data[predicted_attribute] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / total_size, random_state=1) 

print("")
print("Original Training: DecisionTreeClassifier, Parameters: ")
print("-------------")

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?
original_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Original Accuracy:", original_accuracy)

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5), dpi=500)
tree.plot_tree(classifier,
               feature_names=attributes,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('graphics/Original-DecisionTreeClassifier_.png')
print("")
print("Original Training: DecisionTreeClassifier, Parameters: criterion='gini', max_depth=3")
print("-------------")

classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
classifier = classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?
original_accuracy = metrics.accuracy_score(y_test, y_pred)
print("Original Accuracy:", original_accuracy)

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5), dpi=500)
tree.plot_tree(classifier,
               feature_names=attributes,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('graphics/Original-DecisionTreeClassifier_criterion=gini-max_depth=3.png')

print("")
print("Bias Mitigation")
print("-------------")
print("Method: Reweighing")
mitigation_method = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_mitigated_train = mitigation_method.fit_transform(dataset_original_train)

print("")
print("After Mitigation Bias Checking: Sex Bias Checking")
print("-------------")
print("Dataset: Adult Census Income")
metric_mitigated_train = BinaryLabelDatasetMetric(dataset_mitigated_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
explainer_train = MetricTextExplainer(metric_mitigated_train)
			
print("After mitigation explainer: " + explainer_train.disparate_impact())
print("After mitigation statistical_parity_difference: %f" % metric_mitigated_train.statistical_parity_difference())
print("After mitigation mean_difference: %f" % metric_mitigated_train.mean_difference())
	

print("")
print("After Mitigation Training: DecisionTreeClassifier, Parameters: ")
print("-------------")

classifier = DecisionTreeClassifier()
model = make_pipeline(StandardScaler(),
                      classifier)
fit_params = {'decisiontreeclassifier__sample_weight': dataset_mitigated_train.instance_weights}
model_mitigated_train = model.fit(dataset_mitigated_train.features, dataset_mitigated_train.labels.ravel(), **fit_params)

y_pred = model_mitigated_train.predict(dataset_original_test.features)
y_test = dataset_original_test.labels

# Model Accuracy, how often is the classifier correct?
after_mitigation_accuracy = metrics.accuracy_score(y_test, y_pred)
print("After Mitigation Accuracy:", after_mitigation_accuracy)
plt.figure(figsize=(12, 5), dpi=500)
tree.plot_tree(classifier,
               feature_names=attributes,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('graphics/Mitigated-DecisionTreeClassifier_.png')

print("")
print("After Mitigation Training: DecisionTreeClassifier, Parameters: criterion='gini', max_depth=3")
print("-------------")

classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)
model = make_pipeline(StandardScaler(),
                      classifier)
fit_params = {'decisiontreeclassifier__sample_weight': dataset_mitigated_train.instance_weights}
model_mitigated_train = model.fit(dataset_mitigated_train.features, dataset_mitigated_train.labels.ravel(), **fit_params)

y_pred = model_mitigated_train.predict(dataset_original_test.features)
y_test = dataset_original_test.labels

# Model Accuracy, how often is the classifier correct?
after_mitigation_accuracy = metrics.accuracy_score(y_test, y_pred)
print("After Mitigation Accuracy:", after_mitigation_accuracy)
plt.figure(figsize=(12, 5), dpi=500)
tree.plot_tree(classifier,
               feature_names=attributes,
               class_names=["0", "1"],
               filled=True,
               rounded=True);
plt.savefig('graphics/Mitigated-DecisionTreeClassifier_criterion=gini-max_depth=3.png')

