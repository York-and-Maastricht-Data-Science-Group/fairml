'''
    @note: York-Maastricht Project (YMP)
    @organization: University of York, University of Maastricht
    @author: Alfa Yohannis
'''

import inspect
import os.path
import json
import numbers
import tensorflow.compat.v1 as tf
import matplotlib
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from numpy.core._simd import baseline
np.random.seed(0)

from collections import defaultdict
from collections import OrderedDict
from sklearn import tree
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from aif360.metrics import Metric
from aif360.explainers import MetricJSONExplainer
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple

from IPython.display import Markdown, display
from IPython import get_ipython
from aif360.explainers.metric_text_explainer import MetricTextExplainer


def print_message(text):
    if get_ipython() == None:
        print(text)
    else:
        display(Markdown(text))


class FairML():
    
    def __init__(self):
        """
        """
        self.results = {}
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
        
    def get_generic_distortion_for_optimised_preprocessing(self, vold, vnew):
        ''' generic distortion for optimised preprocessing
        '''
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
        self.name = None
        self.fairml = None
        self.results = []
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
        self.dataset_original_valid = None
        self.dataset_original_test = None
        self.train_dataset_module = None
        self.test_dataset_module = None
        self.validation_dataset_module = None
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
    
    def train(self, dataset_train, classifier, without_weight=True , scaler=StandardScaler):
        self.classifiers[classifier.__class__.__name__] = classifier.__class__
        model_train = None
        if without_weight:
            model_train = classifier.fit(dataset_train.features, dataset_train.labels.ravel())
        else:
            model = make_pipeline(scaler(), classifier)
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
    
    def init_new_result(self, mitigation_algorithm_name, mitigation_algorithm_params, dataset_name, classifier_name, parameters):
        self.mitigation_results = defaultdict(list)
        self.results.append(self.mitigation_results)
        if len(mitigation_algorithm_params) > 0:
            self.mitigation_results["Mitigation"].append(mitigation_algorithm_name + "\n" + mitigation_algorithm_params)
        else:
            self.mitigation_results["Mitigation"].append(mitigation_algorithm_name + mitigation_algorithm_params)
            
        self.mitigation_results["Dataset"].append(dataset_name + "(" + str(self.training_size) + ":" + 
                                                   str(self.test_size) + ":" + str(self.validation_size) + ")")
        if len(parameters) > 0:
            self.mitigation_results["Classifier"].append(classifier_name + "\n" + parameters)
        else:
            self.mitigation_results["Classifier"].append(classifier_name + parameters)
        # self.mitigation_results["sklearn_accuracy"].append(accuracy)
        
    def display_summary(self):
        line_num = pd.Series(range(1, len(self.results) + 1))
        self.summary_table = pd.concat([pd.DataFrame(m) for m in self.results], axis=0).set_axis(line_num)
        
        for metric in self.metrics:
            for name, values in self.summary_table [[metric]].iteritems():
                if name == "accuracy":
                    fairest_value, fairest_line = self.get_fairest_value(values, 1)
                    self.fairest_values[name] = fairest_value
                    self.fairest_combinations[name] = fairest_line
                    self.table_colours[name] = self.get_colours(values, 1)
                    self.ideal_values[name] = 1 
                elif name == "balanced_accuracy":
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
            print("")
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
        
        if get_ipython() == None:
            matplotlib.use("TkAgg")
            
        data.plot.bar(figsize=(16, 4), rot=0, xlabel="Bias Mitigation",
                      title="Normalised Metrics (Value 1 Indicates the Bias Mitigation is the Best Option for the Metric)")
        # plot.show(block=True)
        image = "graphics/" + self.name.replace(" ", "_").lower() + ".png"
        
        if get_ipython() == None:
            plot.savefig(image)  
            plot.show(block=False)
        else:
            # backend = matplotlib.get_backend()
            # print(backend)
            # matplotlib.use("Agg")
            plot.savefig(image)
            # matplotlib.use(backend)        
            # display(Image(filename=image))
        
        if get_ipython() == None:
            matplotlib.use("Agg")
    
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
            if ideal_value <= 0:
                val = abs((max_num if values.get(key=i) is None else values.get(key=i)) - ideal_value)
            else:
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
            if ideal_value <= 0:
                x1 = 1
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
                    if ideal_value <= 0:
                        x1 = 1
                    else:
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
                b = 0
                if num <= 255: 
                    r = "{0:0{1}x}".format(255, 2)
                    g = "{0:0{1}x}".format(255, 2)
                    b = "{0:0{1}x}".format(255 - num, 2)
                else:
                    r = "{0:0{1}x}".format(255 - (num - 255), 2)
                    g = "{0:0{1}x}".format(255, 2)
                    b = "{0:0{1}x}".format(0, 2)
                    
                cell_formats[index] = bold + 'background-color: #' + r + g + b + ';'
                # print(num)
                # print(cell_formats[index])
        
        return cell_formats 
    
    def get_prediction_probability(self, model, dataset_orig, scaled_target_dataset_features):
        fav_idx = np.where(model.classes_ == dataset_orig.favorable_label)[0][0]
        y_test_pred_prob = model.predict_proba(scaled_target_dataset_features)[:, fav_idx]
        return y_test_pred_prob
    
    def find_optimal_threshold_metric(self, metric_name, baseline_dataset, predicted_dataset,
                                         unprivileged_groups=None, privileged_groups=None,
                                         start=0.01, end=1.00, num_thresh=100, model=None
                                         ):
        try:
            # this is for sklearn classifier
            fav_idx = np.where(model.classes_ == baseline_dataset.favorable_label)[0][0]
            predicted_dataset.scores = model.predict_proba(predicted_dataset.features)[:, fav_idx] 
        except AttributeError:
            # this is for aif360 inprocessing algorithm
            fav_idx = 0
            predicted_dataset.scores = model.predict(predicted_dataset).scores
        
        unprivileged_groups = self.unprivileged_groups if unprivileged_groups is None else unprivileged_groups 
        privileged_groups = self.privileged_groups if privileged_groups is None else privileged_groups
        
        balanced_accuracy_arr = []
        metric_val_arr = []
        metric_obj_arr = []
        thresh_arr = np.linspace(start, end, num_thresh)
        for idx, class_thresh in enumerate(thresh_arr):
            
            fav_inds = predicted_dataset.scores > class_thresh
            predicted_dataset.labels[fav_inds] = predicted_dataset.favorable_label
            predicted_dataset.labels[~fav_inds] = predicted_dataset.unfavorable_label
            
            classified_metric_orig_valid = FairMLMetric(baseline_dataset,
                                                     predicted_dataset,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
            
            balanced_accuracy_arr.insert(idx, getattr(classified_metric_orig_valid, "balanced_accuracy")())
            metric_val_arr.insert(idx, getattr(classified_metric_orig_valid, metric_name)())
            metric_obj_arr.insert(idx, classified_metric_orig_valid)
        
        max_value = max(balanced_accuracy_arr)
        best_ind = balanced_accuracy_arr.index(max_value)
        best_thresh = thresh_arr[best_ind]
        best_metric = metric_obj_arr[best_ind]
        
        return best_metric, balanced_accuracy_arr, metric_val_arr, thresh_arr, best_thresh
    
    def measure_bias(self, metric_name, baseline_dataset=None, predicted_dataset=None,
         privileged_groups=None, unprivileged_groups=None, optimal_threshold=False, model=None,
         plot_threshold=False, **params):
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
        
        if optimal_threshold:
            metric_mitigated_train, balanced_accuracy_arr, metric_val_arr, thresh_arr, best_thresh = self. \
                find_optimal_threshold_metric(metric_name, baseline_dataset=baseline_dataset,
                predicted_dataset=predicted_dataset, unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups, model=model)
            
            if plot_threshold:
                fig, ax1 = plot.subplots(figsize=(16, 4))
                ax1.plot(thresh_arr, balanced_accuracy_arr)
                ax1.set_xlabel('classification thresholds', fontsize=16, fontweight='bold')
                ax1.set_ylabel('balanced accuracy', color='b', fontsize=16, fontweight='bold')
                ax1.xaxis.set_tick_params(labelsize=14)
                ax1.yaxis.set_tick_params(labelsize=14)
                ax1.set_title("best balanced accuracy: {:.6f}, {}: {:.6f}, best_threshold: {:.6f}". \
                    format(metric_mitigated_train.balanced_accuracy(), metric_name.replace("_", " "),
                        getattr(metric_mitigated_train, metric_name)(), best_thresh))
                
                ax2 = ax1.twinx()
                ax2.plot(thresh_arr, metric_val_arr, color='r')
                ax2.set_ylabel(metric_name.replace("_", " "), color='r', fontsize=16, fontweight='bold')
                ax2.axvline(best_thresh, color='k', linestyle=':')
                ax2.yaxis.set_tick_params(labelsize=14)
                ax2.grid(True)
                
                image = "graphics/balanced_accuracy({:.3f})_vs_{}({:.3f})_{}.png". \
                    format(metric_mitigated_train.balanced_accuracy(), metric_name,
                        getattr(metric_mitigated_train, metric_name)(), best_thresh)
            
                if get_ipython() is None:  # on console app
                    plot.savefig(image) 
                    plot.show(block=False) 
                else:  # on jupyter notebook
                    plot.savefig(image)

        else:
            if predicted_dataset is not None:
                metric_mitigated_train = FairMLMetric(baseline_dataset,
                                                     predicted_dataset,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
            else:
                metric_mitigated_train = BinaryLabelDatasetMetric(baseline_dataset,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
            
        if metric_name == "rich_subgroup":
            if predicted_dataset is not None:
                params["predictions"] = array_to_tuple(predicted_dataset.labels)
            else:
                self.mitigation_results[metric_name].append(1.0)
                return    
        
        explainer_train = MetricFairMLExplainer(metric_mitigated_train)
        
        if not callable(getattr(metric_mitigated_train, metric_name, None)):
            self.mitigation_results[metric_name].append(None)
        else: 
            explanation = json.loads(getattr(explainer_train, metric_name)(**params), object_pairs_hook=OrderedDict)
            
            if get_ipython() == None:
                print("")
                for key in explanation:
                    print(key[0].upper() + key[1:] + ": " + str(explanation[key]))
            else:
                text = ""
                for key in explanation:
                    text += "**" + key[0].upper() + key[1:] + ":** " + str(explanation[key]) + "<br/>"     
                display(Markdown(text))

            self.mitigation_results[metric_name].append(getattr(metric_mitigated_train, metric_name)(**params))
    
    
class MetricTextFairMLExplainer(MetricTextExplainer):

    def __init__(self, metric):
        """Initialize a `MetricTextExplainer` object.

        Args:
            metric (Metric): The metric to be explained.
        """
        if isinstance(metric, Metric):
            self.metric = metric
        else:
            raise TypeError("metric must be a Metric.")
    
    def generalized_false_negative_rate_difference(self):
        ''' Unprivileged Generated False Negative Rate (GFNR) - Privileged Generated False Negative Rate (GFNR)
        '''
        return "Generalized False Negative Rate (GFNR) difference: {}".format(self.metric.generalized_false_negative_rate(False) - self.metric.generalized_false_negative_rate(True))
    
    def generalized_false_positive_rate_difference(self):
        ''' Unprivileged Generated False Positive Rate (GFPR) - Privileged Generated False Positive Rate (GFPR)
        '''
        return "Generalized False Positive Rate (GFPR) difference: {}".format(self.metric.generalized_false_positive_rate(False) - self.metric.generalized_false_positive_rate(True))  
    
    def balanced_accuracy(self, privileged=None):
        if privileged is None:
            return "Balanced Classification accuracy (BACC): {}".format(
                self.metric.balanced_accuracy(privileged=privileged))
        return "Balanced Classification accuracy on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.balanced_accuracy(privileged=privileged))

    def rich_subgroup(self, predictions, fairness_def='FP'):
        return "Gamma disparity with respect to the fairness_def: {}".format(
            self.metric.rich_subgroup(predictions, fairness_def))
    

class MetricFairMLExplainer(MetricTextFairMLExplainer, MetricJSONExplainer):
    '''
    The class is derived from MetricJSONExplainer because I think there is a logic error
    in disparate_impact() and  statistical_parity_difference().
    The line if isinstance(self.metric, BinaryLabelDatasetMetric): should be
    if type(self.metric) is BinaryLabelDatasetMetric: otherwise any class derived from
    BinaryLabelDatasetMetric, including ClassificationMetric 
    always goes to the first condition.
    '''

    def generalized_false_negative_rate_difference(self):
        outcome = super(MetricFairMLExplainer, self).generalized_false_negative_rate_difference()
        response = OrderedDict((
            ("metric", "Generalized False Negative Rate (GFNR):"),
            ("message", outcome),
            ("generalizedFalseNegativeRateUnprivileged", self.metric.generalized_false_negative_rate(privileged=False)),
            ("generalizedFalseNegativeRatePrivileged", self.metric.generalized_false_negative_rate(privileged=True)),
            ("description", "Unprivileged Generated False Negative Rate (GFNR) - Privileged Generated False Negative Rate (GFNR)."),
            ("ideal", " The ideal value of this metric is 0")
        ))
        return json.dumps(response)
    
    def generalized_false_positive_rate_difference(self):
        outcome = super(MetricFairMLExplainer, self).generalized_false_positive_rate_difference()
        response = OrderedDict((
            ("metric", "Generalized False Positive Rate (GFPR):"),
            ("message", outcome),
            ("generalizedFalsePositiveRateUnprivileged", self.metric.generalized_false_positive_rate(privileged=False)),
            ("generalizedFalsePositiveRatePrivileged", self.metric.generalized_false_positive_rate(privileged=True)),
            ("description", "Unprivileged Generated False Positive Rate (GFPR) - Privileged Generated False Positive Rate (GFPR)."),
            ("ideal", " The ideal value of this metric is 0")
        ))
        return json.dumps(response)
    
    def disparate_impact(self):
        outcome = super(MetricJSONExplainer, self).disparate_impact()
        response = []
        if type(self.metric) is BinaryLabelDatasetMetric:
            response = OrderedDict((
                ("metric", "Disparate Impact"),
                ("message", outcome),
                ("numPositivePredictionsUnprivileged", self.metric.num_positives(privileged=False)),
                ("numUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivePredictionsPrivileged", self.metric.num_positives(privileged=True)),
                ("numPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the ratio of rate of favorable outcome for the unprivileged group to that of the privileged group."),
                ("ideal", "The ideal value of this metric is 1.0 A value < 1 implies higher benefit for the privileged group and a value >1 implies a higher benefit for the unprivileged group.")
            ))
        else:
            response = OrderedDict((
                ("metric", "Disparate Impact"),
                ("message", outcome),
                ("numPositivePredictionsUnprivileged", self.metric.num_pred_positives(privileged=False)),
                ("numUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivePredictionsPrivileged", self.metric.num_pred_positives(privileged=True)),
                ("numPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the ratio of likelihood of favorable outcome for the unprivileged group to that of the privileged group."),
                ("ideal", "The ideal value of this metric is 1.0")
            ))
        return json.dumps(response)
    
    def statistical_parity_difference(self):
        outcome = super(MetricJSONExplainer, self).statistical_parity_difference()
        response = []
        if type(self.metric) is BinaryLabelDatasetMetric:
            response = OrderedDict((
                ("metric", "Statistical Parity Difference"),
                ("message", outcome),
                ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
                ("numInstancesUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
                ("numInstancesPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the difference of the rate of favorable outcomes received by the unprivileged group to the privileged group."),
                ("ideal", " The ideal value of this metric is 0")
            ))
        else:
            response = OrderedDict((
                ("metric", "Statistical Parity Difference"),
                ("message", outcome),
                ("numPositivesUnprivileged", self.metric.num_pred_positives(privileged=False)),
                ("numInstancesUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivesPrivileged", self.metric.num_pred_positives(privileged=True)),
                ("numInstancesPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the difference of the rate of favorable outcomes received by the unprivileged group to the privileged group."),
                ("ideal", " The ideal value of this metric is 0")
            ))
        return json.dumps(response)

    def balanced_accuracy(self):
        outcome = super(MetricFairMLExplainer, self).balanced_accuracy()
        response = OrderedDict((
            ("metric", "Balanced Accuracy"),
            ("message", outcome),
            ("truePositiveRateUnprivileged", self.metric.true_positive_rate(privileged=False)),
            ("trueNegativeRateUnprivileged", self.metric.true_negative_rate(privileged=False)),
            ("truePositiveRatePrivileged", self.metric.true_positive_rate(privileged=True)),
            ("trueNegativeRatePrivileged", self.metric.true_negative_rate(privileged=True)),
            ("description", "Return the balanced accuracy of the ratios of true positives and true negatives examples in the dataset, :math:`0.5 * (TPR + TNR)`,  optionally conditioned on protected attributes."),
            ("ideal", " The ideal value of this metric is 1")
        ))
        return json.dumps(response)
    
    def rich_subgroup(self, predictions, fairness_def='FP'):
        outcome = super(MetricFairMLExplainer, self).rich_subgroup(predictions, fairness_def)
        response = OrderedDict((
            ("metric", "Rich Subgroup/Gamma Disparity"),
            ("message", outcome),
            ("description", "Audit dataset with respect to rich subgroups defined by linear thresholds of sensitive attributes. " + 
                "fairness_def is 'FP' or 'FN' for rich subgroup wrt to false positive or false negative rate. " + 
                "predictions is a hashable tuple of predictions. Typically the labels attribute of a GerryFairClassifier."),
            ("ideal", " The ideal value of this metric is 0")
        ))
        return json.dumps(response)
    

class FairMLMetric(ClassificationMetric):
    '''Extend the Classification Metric to Include Balanced Accuracy
    '''
    
    def generalized_false_negative_rate_difference(self):
        ''' Unprivileged Generated False Negative Rate (GFNR) - Privileged Generated False Negative Rate (GFNR)
        '''
        return self.generalized_false_negative_rate(False) - self.generalized_false_negative_rate(True)
    
    def generalized_false_positive_rate_difference(self):
        ''' Unprivileged Generated False Positive Rate (GFPR) - Privileged Generated False Positive Rate (GFPR)
        '''
        return self.generalized_false_positive_rate(False) - self.generalized_false_positive_rate(True)  
    
    def balanced_accuracy(self, privileged=None):
        """Return the balanced accuracy of the ratio of true positives and 
        true negatives examples in the dataset, :math:`0.5 * (TPR + TNR)`, 
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` 
                must be provided at initialization to condition on them.
        """
        return 0.5 * (self.true_positive_rate(privileged) + self.true_negative_rate(privileged))
        
