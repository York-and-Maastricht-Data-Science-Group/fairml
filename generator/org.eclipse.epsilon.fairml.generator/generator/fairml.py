import inspect
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
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from IPython.display import Markdown, display
from IPython import get_ipython


def is_preprocessing(class_name):
    return 'preprocessing' in class_name.__file__


def is_inprocessing(class_name):
    return 'inprocessing' in class_name.__file__


def is_postprocessing(class_name):
    return 'postprocessing' in class_name.__file__

    
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
        self.optim_options = {
            "distortion_fun": self.get_generic_distortion_for_optimised_preprocessing,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        
        dir_name = 'graphics'
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        
        dir_name = 'data'
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            os.mkdir(dir_name)    
        
    ''' generic distortion for optimised preprocessing
    '''
    def get_generic_distortion_for_optimised_preprocessing(self, vold, vnew):
        return 1.0  
            
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
        self.dataset_original_validation = None
        self.dataset_original_test = None
        self.privileged_groups = None
        self.unprivileged_groups = None
        self.mitigation_results = None
        self.metrics = ['Accuracy']
        self.fairest_values = {}
        self.fairest_combinations = {}
        self.summary_table = None
        
    
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
    
    def create_mitigation_method(self, mitigation_class, **params):
        signatures = inspect.signature(mitigation_class)
        if 'privileged_groups' and 'unprivileged_groups' in signatures.parameters:
            params['privileged_groups'] = self.privileged_groups
            params['unprivileged_groups'] = self.unprivileged_groups
        
        if 'optimizer' in signatures.parameters:
            params['optimizer'] = OptTools
        
        if 'optim_options' in signatures.parameters:
            params['optim_options'] = self.fairml.optim_options
                
        mitigation_method = mitigation_class(**params)
        return mitigation_method
         
    
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
        if not metric_name in self.metrics:      
            self.metrics.append(metric_name)
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
        self.summary_table = pd.concat([pd.DataFrame(m) for m in self.fairml.results], axis=0).set_axis(line_num)
        
        for metric in self.metrics:
            for name, values in self.summary_table [[metric]].iteritems():
                if name == "Accuracy":
                    fairest_value, fairest_line = self.get_fairest_value(values, 1)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                elif name == "disparate_impact":
                    fairest_value, fairest_line = self.get_fairest_value(values, 1)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                elif name == "statistical_parity_difference":
                    fairest_value, fairest_line = self.get_fairest_value(values, 0)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
        
        if get_ipython() == None:
            print("Original Data size: " + str(len(self.dataset_original.instance_names))) 
            print("Predicted attribute: " + self.predicted_attribute)
            print("Protected attributes: " + ", ".join(self.protected_attributes)) 
            print("Favorable classes: " + str(self.favorable_class)) 
            print("Dropped attributes: " + ", ".join(self.dropped_attributes))
            print("Training data size (ratio): " + str(self.training_size)) 
            print("Test data size (ratio): " + str(self.test_size))
            print("")
            print(self.summary_table )
    
        else:
            display(Markdown("Original Data size: " + str(len(self.dataset_original.instance_names)) + "</br>" + 
                "Predicted attribute: " + self.predicted_attribute + "</br>" + 
                "Protected attributes: " + ", ".join(self.protected_attributes) + "</br>" + 
                "Favourable classes: " + str(self.favorable_class) + "</br>" + 
                "Dropped attributes: " + ", ".join(self.dropped_attributes) + " </br>"
                "Training data size (ratio): " + str(self.training_size) + "</br>" + 
                "Test data size (ratio): " + str(self.test_size)))
        
        
        return self.summary_table  
    
    def get_fairest_value(self, values, ideal_value):
        fairest_value = None
        fairest_combination = 1
        x1 = values.get(key = fairest_combination) - ideal_value
        min_abs_val = abs(x1)
        min_val_sign = ""
        if x1 < 0:
            min_val_sign = "-"
        elif x1 >= 0:
            min_val_sign = "+"
        
        if len(values) > 1:
            i = 1
            for value in values:
                i = i + 1
                x1 = value - ideal_value 
                temp = abs(x1)
                if temp < min_abs_val:
                    fairest_combination = i
                    min_abs_val = temp
                    if x1 < 0:
                        min_val_sign = "-"
                    elif x1 >= 0:    
                        min_val_sign = "+"
        
        if min_val_sign == "-":
            fairest_value =(-1 * min_abs_val) + ideal_value
        else:
            fairest_value = (1 * min_abs_val) + ideal_value
        
        return fairest_value, fairest_combination    

       
    def highlight_fairest_values(self, row):
        cell_formats = [''] * len(row)
        for metric in self.fairest_combinations:
            if row.name == self.fairest_combinations[metric]:
                index = self.summary_table.columns.get_loc(metric)
                cell_formats[index] = 'font-weight: bold; background-color: #e6ffe6;'
        
        return cell_formats 
  

