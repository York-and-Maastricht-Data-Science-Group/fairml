import pandas as pd
import matplotlib.pyplot as plt
import os.path 

from collections import defaultdict
from sklearn import tree
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.explainers import MetricTextExplainer
from IPython.display import Markdown, display
from IPython import get_ipython


def print_message(text):
    if get_ipython() == None:
        print(text)
    else:
        display(Markdown(text))


class FairML():
    
    
    def __init__(self):
        """
        """
        self.results = []
        self.line_num_counter = 1
        self.bias_mitigations = []
        
        dir_name = 'graphics'
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        
        dir_name = 'data'
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)    
          
            
    def add_bias_mitigation(self, bias_mitigation):
        self.bias_mitigations.append(bias_mitigation)
        bias_mitigation.fairml = self
        return bias_mitigation

        
class BiasMitigation():
    """
        Main Class for BiasMitigation
    """

    def __init__(self):
        """
        Args: None
        """
        self.fairml = None
        self.predicted_attribute = None
        self.protected_attributes = []
        self.favorable_class = 1
        self.privileged_class = 1
        self.unprivileged_class = 0
        self.dropped_attributes = []
        self.na_values = []
        self.training_size = 7    
        self.test_size = 3 
        self.total_size = 7 + 3
        self.categorical_features = []
        self.default_mappings = None
        self.resource = None
        self.data = None
        self.dataset_original = None
        self.dataset_original_train = None
        self.dataset_original_test = None
        self.privileged_groups = None
        self.unprivileged_groups = None
        self.mitigation_results = None


    def check_accuracy(self, model, dataset_test):
        y_pred = model.predict(dataset_test.features)
        y_test = dataset_test.labels.ravel()
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return accuracy


    def create_predicted_dataset(self, dataset, model):
        """
        Args:
            dataset: the dataset that contains labels to be predicted
            model: the model to make prediction 
        
        Returns:
            Predicted dataset
        """
        dataset_predicted = dataset.copy()
        y_val_pred = model.predict(dataset.features)
        dataset_predicted.labels = y_val_pred
        return dataset_predicted
    
    
    def train(self, dataset_train, classifier):
        # classifier = DecisionTreeClassifier(criterion='gini', max_depth=7)
        model = make_pipeline(StandardScaler(), classifier)
        name = type(classifier).__name__.lower()
        fit_params = {name + '__sample_weight': dataset_train.instance_weights}
        model_train = model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        return model_train
    
    
    def drawModel(self, classifier, dataset, filename):
        if isinstance(classifier, DecisionTreeClassifier): 
            plt.figure(figsize=(12, 5), dpi=500)
            tree.plot_tree(classifier,
                           feature_names=dataset.feature_names,
                           # class_names=["1", "0"],
                           filled=True,
                           rounded=True);
            plt.savefig(filename)
    
    
    def measure_bias(self, metric_name, dataset, predicted_dataset,
                     privileged_groups, unprivileged_groups):
        """Compute the number of true/false positives/negatives, optionally
        conditioned on protected attributes.
        
        Args:
            metric_name (String): The name of the metric to be called.
            
        Returns:
            None
        """
        metric_mitigated_train = ClassificationMetric(dataset,
                                             predicted_dataset,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
        explainer_train = MetricTextExplainer(metric_mitigated_train)
        print_message("")        
        getattr(metric_mitigated_train, metric_name)()
        print_message("After mitigation " + metric_name + ": %f" % getattr(metric_mitigated_train, metric_name)())
        print_message("After mitigation explainer: " + getattr(explainer_train, metric_name)())
        self.mitigation_results[metric_name].append(getattr(metric_mitigated_train, metric_name)())
    
    
    def init_new_result(self, mitigation_algorithm_name, dataset_name, classifier_name, accuracy):
        self.mitigation_results = defaultdict(list)
        self.fairml.results.append(self.mitigation_results)
        self.mitigation_results["Mitigation"].append(mitigation_algorithm_name)
        self.mitigation_results["Dataset"].append(dataset_name + "(" + str(self.training_size) + ":" + str(self.test_size) + ")")
        self.mitigation_results["Classifier"].append(classifier_name)
        self.mitigation_results["Accuracy"].append(accuracy)
        
        
    def display_summary(self):
        print("")
        line_num = pd.Series(range(1, len(self.fairml.results) + 1))
        table = pd.concat([pd.DataFrame(m) for m in self.fairml.results], axis=0).set_axis(line_num)
        
        if get_ipython() == None:
            print("Original Data size: " + str(len(self.dataset_original.instance_names))) 
            print("Predicted attribute: " + self.predicted_attribute)
            print("Protected attributes: " + ", ".join(self.protected_attributes)) 
            print("Favorable classes: " + str(self.favorable_class)) 
            print("Dropped attributes: " + ", ".join(self.dropped_attributes))
            print("Training data size (ratio): " + str(self.training_size)) 
            print("Test data size (ratio): " + str(self.test_size))
            print("")
            print(table)
        else:
            display(Markdown("Original Data size: " + str(len(self.dataset_original.instance_names)) + "</br>" + 
                "Predicted attribute: " + self.predicted_attribute + "</br>" + 
                "Protected attributes: " + ", ".join(self.protected_attributes) + "</br>" + 
                "Favorable classes: " + str(self.favorable_class) + "</br>" + 
                "Dropped attributes: " + ", ".join(self.dropped_attributes) + " </br>"
                "Training data size (ratio): " + str(self.training_size) + "</br>" + 
                "Test data size (ratio): " + str(self.test_size)))
                    
        return table 
