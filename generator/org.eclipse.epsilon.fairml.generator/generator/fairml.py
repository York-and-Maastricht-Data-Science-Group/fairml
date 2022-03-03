import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import os.path
import json 
import numbers

from collections import defaultdict
from collections import OrderedDict
from sklearn import tree
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.explainers import MetricJSONExplainer
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from IPython.display import Markdown, display, JSON, display_json
from IPython import get_ipython

import tensorflow.compat.v1 as tf
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from IPython.core.pylabtools import figsize


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
        self.training_size = 5
        self.validation_size = 3     
        self.test_size = 2 
        self.total_size = self.training_size + self.validation_size + self.test_size
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
        self.metrics = []
        self.fairest_values = {}
        self.fairest_combinations = {}
        self.table_colours = {}
        self.summary_table = None
        self.tf_sessions = []
        self.mitigation_algorithms = {}
        self.classifiers = {}
        self.ideal_values = {}
    
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
        dataset_predicted = dataset.copy(deepcopy=True)
        y_val_pred = model.predict(dataset.features)
        dataset_predicted.labels = y_val_pred
        return dataset_predicted
    
    def get_prediction_probability(self, model, dataset_orig, scaled_target_dataset_features):
        fav_idx = np.where(model.classes_ == dataset_orig.favorable_label)[0][0]
        y_test_pred_prob = model.predict_proba(scaled_target_dataset_features)[:, fav_idx]
        return y_test_pred_prob
        
    def create_mitigation_method(self, mitigation_class, **params):
        self.mitigation_algorithms[mitigation_class.__name__] = mitigation_class
        signatures = inspect.signature(mitigation_class)
        if 'privileged_groups' and 'unprivileged_groups' in signatures.parameters:
            if 'privileged_groups' not in params:
                params['privileged_groups'] = self.privileged_groups
            if 'unprivileged_groups' not in params:
                params['unprivileged_groups'] = self.unprivileged_groups

        # if 'art_classifier' in signatures.parameters:
        #     if 'art_classifier' not in params:
        #         params['art_classifier'] =  Classifier()
                
        if 'estimator' in signatures.parameters:
            if 'estimator' not in params:
                params['estimator'] = LogisticRegression(solver='lbfgs')
        
        if 'constraints' in signatures.parameters:
            if 'constraints' not in params:
                params['constraints'] = "DemographicParity"
            # check import fairlearn.reductions 
            # "DemographicParity", "TruePositiveRateDifference", "ErrorRateRatio."
            
        if 'optimizer' in signatures.parameters:
            if 'optimizer' not in params:
                params['optimizer'] = OptTools
        
        if 'optim_options' in signatures.parameters:
            if 'optim_options' not in params:
                params['optim_options'] = self.fairml.optim_options
        
        if 'scope_name' in signatures.parameters:
            if 'scope_name' not in params:
                params['scope_name'] = 'debiased_classifier'
            
        if 'sess' in signatures.parameters:
            if 'sess' not in params:
                tf.reset_default_graph()
                tf.disable_eager_execution()
                sess = tf.Session()
                self.tf_sessions.append(sess)
                params['sess'] = sess
            
        mitigation_method = mitigation_class(**params)
        
        return mitigation_method
    
    def train(self, dataset_train, classifier):
        self.classifiers[classifier.__class__.__name__] = classifier.__class__
        model = make_pipeline(StandardScaler(), classifier)
        name = type(classifier).__name__.lower()
        fit_params = {name + '__sample_weight': dataset_train.instance_weights}
        model_train = model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)
        return model_train
    
    def drawModel(self, classifier, dataset, filename):
        if isinstance(classifier, DecisionTreeClassifier): 
            plot.figure(figsize=(12, 5), dpi=500)
            tree.plot_tree(classifier,
                           feature_names=dataset.feature_names,
                           # class_names=["1", "0"],
                           filled=True,
                           rounded=True);
            plot.savefig(filename)
    
    def measure_bias(self, metric_name, dataset, predicted_dataset=None,
                     privileged_groups=None, unprivileged_groups=None):
        """Compute the number of true/false positives/negatives, optionally
        conditioned on protected attributes.
        
        Args:
            metric_name (String): The name of the metric to be called.
            
        Returns:
            None
        """
        if not metric_name in self.metrics: 
            self.metrics.append(metric_name)
        
        metric_mitigated_train = None
        if predicted_dataset is not None:
            metric_mitigated_train = ClassificationMetric(dataset,
                                                 predicted_dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        else:
            metric_mitigated_train = BinaryLabelDatasetMetric(dataset,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    
        explainer_train = MetricJSONExplainer(metric_mitigated_train)
        
        if not callable(getattr(metric_mitigated_train, metric_name, None)):
            # print_message("After mitigation " + metric_name + ": %f" % 0)
            # print_message("After mitigation: " + str(None))
            self.mitigation_results[metric_name].append(None)
        else: 
            explanation = json.loads(getattr(explainer_train, metric_name)(), object_pairs_hook=OrderedDict)
            
            if get_ipython() == None:
                print("")
                for key in explanation:
                    print(key[0].upper() + key[1:] + ": " + str(explanation[key]))
            else:
                text = ""
                for key in explanation:
                    text += "**" + key[0].upper() + key[1:] + ":** " + str(explanation[key]) + "<br/>"     
                display(Markdown(text))
            # print_message("")        
            # getattr(metric_mitigated_train, metric_name)()
            # a = float(0.5) * (metric_mitigated_train.true_positive_rate() + metric_mitigated_train.true_negative_rate())
            # print_message("After mitigation Balanced accuracy: %f" % a )
            # print_message("After mitigation " + metric_name + ": %f" % getattr(metric_mitigated_train, metric_name)())
            # print_message("After mitigation explainer: " + getattr(explainer_train, metric_name)())
            # print_message("After mitigation: " + getattr(explainer_train, metric_name)())
            self.mitigation_results[metric_name].append(getattr(metric_mitigated_train, metric_name)())
    
    def init_new_result(self, mitigation_algorithm_name, dataset_name, classifier_name, parameters):
        self.mitigation_results = defaultdict(list)
        self.fairml.results.append(self.mitigation_results)
        self.mitigation_results["Mitigation"].append(mitigation_algorithm_name)
        self.mitigation_results["Dataset"].append(dataset_name + "(" + str(self.training_size) + ":" + 
                                                   str(self.test_size) + ":" + str(self.validation_size) + ")")
        if len(parameters) > 0:
            self.mitigation_results["Classifier"].append(classifier_name + "\n" + parameters)
        else:
            self.mitigation_results["Classifier"].append(classifier_name + parameters)
        # self.mitigation_results["sklearn_accuracy"].append(accuracy)
        
    def display_summary(self):
        print("")
        line_num = pd.Series(range(1, len(self.fairml.results) + 1))
        self.summary_table = pd.concat([pd.DataFrame(m) for m in self.fairml.results], axis=0).set_axis(line_num)
        
        for metric in self.metrics:
            for name, values in self.summary_table [[metric]].iteritems():
                if name == "accuracy":
                    fairest_value, fairest_line = self.get_fairest_value(values, 1)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                    self.table_colours[name] = self.get_colours(values, 1)
                    self.ideal_values[name] = 1 
                elif name == "disparate_impact":
                    fairest_value, fairest_line = self.get_fairest_value(values, 1)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                    self.table_colours[name] = self.get_colours(values, 1)
                    self.ideal_values[name] = 1
                elif name == "statistical_parity_difference":
                    fairest_value, fairest_line = self.get_fairest_value(values, 0)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                    self.table_colours[name] = self.get_colours(values, 0)
                    self.ideal_values[name] = 0
                elif "ratio" in name:
                    fairest_value, fairest_line = self.get_fairest_value(values, 1)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                    self.table_colours[name] = self.get_colours(values, 1)
                    self.ideal_values[name] = 1
                else:
                    fairest_value, fairest_line = self.get_fairest_value(values, 0)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                    self.table_colours[name] = self.get_colours(values, 0)
                    self.ideal_values[name] = 0
        
        if get_ipython() == None:
            print("Original Data size: " + str(len(self.dataset_original.instance_names))) 
            print("Predicted attribute: " + self.predicted_attribute)
            print("Protected attributes: " + ", ".join(self.protected_attributes)) 
            print("Favorable classes: " + str(self.favorable_class)) 
            print("Dropped attributes: " + ", ".join(self.dropped_attributes))
            print("Training data size (ratio): " + str(self.training_size)) 
            print("Test data size (ratio): " + str(self.test_size))
            print("Validation data size (ratio): " + str(self.validation_size)) 
            print("")
            display(self.summary_table.to_string())
    
        else:
            display(Markdown("**Description:**<br/>" + 
                "Original Data size: " + str(len(self.dataset_original.instance_names)) + "</br>" + 
                "Predicted attribute: " + self.predicted_attribute + "</br>" + 
                "Protected attributes: " + ", ".join(self.protected_attributes) + "</br>" + 
                "Favourable classes: " + str(self.favorable_class) + "</br>" + 
                "Dropped attributes: " + ", ".join(self.dropped_attributes) + " </br>"
                "Training data size (ratio): " + str(self.training_size) + "</br>" + 
                "Test data size (ratio): " + str(self.test_size) + "</br>" + 
                "Validation data size (ratio): " + str(self.validation_size)))
            
        self.print_explanation()
        
        for session in self.tf_sessions:
            session.close()
        self.tf_sessions.clear()
        
        return self.summary_table  
    
    def display_barchart(self):
        data = self.summary_table.copy(True)
        cols = list(data.columns)
        for col in cols:
            if col not in self.metrics:
                del data[col]
        
        for metric in self.metrics:
            v_min = data[metric].min()
            v_ideal = self.ideal_values[metric]
            data[metric] = data[metric].apply(lambda x: abs((x if isinstance(x, numbers.Number) else v_min) - v_ideal))
            
            v_max = data[metric].max()
            v_min = data[metric].min()
            v_range = v_max - v_min
            data[metric] = data[metric].apply(lambda x: 1 - ((x if isinstance(x, numbers.Number) else v_min) - v_min) / v_range)
            
        data.plot.bar( figsize=(16, 5), rot=0,  xlabel="Bias Mitigation",
                      title="Normalised Metrics (Value 1 Means the Bias Mitigation is the Best Option for the Metric)")
        plot.show(block=True);
    
    def print_explanation(self):
        if get_ipython() == None:
            pass
            # for classifier in self.classifiers:              
            #     text = classifier.__class__.__name__  + ":\n"   
            #     text += classifier.__class__.__doc__ + "\n"  
            #
            #     print(text)
        else:
            for key in self.classifiers:
                text = "**" + self.classifiers[key].__name__ + ":**"            
                text += "<details>" 
                # text += "<summary><i>Click here to show description</i></summary>"  
                text += self.classifiers[key].__doc__.replace("\n", "<br/>") 
                text += "</details>"
                
                display(Markdown(text)) 
                
            for key in self.mitigation_algorithms:
                text = "**" + self.mitigation_algorithms[key].__name__ + ":**"            
                text += "<details>" 
                # text += "<summary><i>Click here to show description</i></summary>"   
                text += self.mitigation_algorithms[key].__doc__.replace("\n", "<br/>")   
                text += "</details>"
                
                display(Markdown(text)) 
            
    def get_colours(self, values, ideal_value):
        max_num = abs((0 if values.get(key=1) is None else values.get(key=1)) - ideal_value)
        min_num = abs((0 if values.get(key=1) is None else values.get(key=1)) - ideal_value) 
        for i in range(1, values.size + 1):
            val = abs((0 if values.get(key=i) is None else values.get(key=i)) - ideal_value)
            if val < min_num:
                min_num = val
            if val > max_num:
                max_num = val
        colours = []
        for i in range(1, values.size + 1):
            val = abs((0 if values.get(key=i) is None else values.get(key=i)) - ideal_value)
            result = 0
            if ((max_num - min_num) != 0) and not pd.isna(val):
                # print(val, min_num, max_num)
                result = int(510.0 - ((val - min_num) / (max_num - min_num) * 510.0))
            colours.append(result) 
        return colours
    
    def get_fairest_value(self, values, ideal_value):
        fairest_value = -1
        fairest_combination = 1
        x1 = 0
        if not values.get(key=fairest_combination) is None: 
            x1 = values.get(key=fairest_combination) - ideal_value
        else:
            x1 = x1 - ideal_value
        # x1 = 0 - ideal_value;
        min_abs_val = abs(x1)
        min_val_sign = ""
        if x1 < 0:
            min_val_sign = "-"
        elif x1 >= 0:
            min_val_sign = "+"
        
        if len(values) > 1:
            i = 0
            for value in values:
                i = i + 1
                if value is None:
                    x1 = 0 - ideal_value
                else:
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
            fairest_value = (-1 * min_abs_val) + ideal_value
        else:
            fairest_value = (1 * min_abs_val) + ideal_value
        
        return fairest_value, fairest_combination    
       
    def highlight_fairest_values(self, row):
        cell_formats = [''] * len(row)
        for metric in self.fairest_combinations:
                bold = ''
                if row.name == self.fairest_combinations[metric]:
                    bold = 'font-weight: bold;'
                index = self.summary_table.columns.get_loc(metric)
                num = self.table_colours[metric][int(row.name) - 1]
                r = 0
                g = 0
                if num <= 255: 
                    r = "{0:0{1}x}".format(255, 2)
                    g = "{0:0{1}x}".format(num, 2)
                else:
                    r = "{0:0{1}x}".format(255 - (num - 255), 2)
                    g = "{0:0{1}x}".format(255, 2)
                    
                cell_formats[index] = bold + 'background-color: #' + r + g + '00;'
                # print(num)
                # print(cell_formats[index])
        
        return cell_formats 

