'''
Created on 1 Mar 2022

@author: Alfa Yohannis
'''
import sys
sys.path.insert(0, '../test-model')

import unittest


class Test(unittest.TestCase):

    tolerance = 0.1
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tutorial_credit_scoring(self):
        import numpy as np
        np.random.seed(0)
        
        from aif360.datasets import GermanDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.algorithms.preprocessing import Reweighing
        
        from IPython.display import Markdown, display
        
        dataset_orig = GermanDataset(
            protected_attribute_names=['age'],  # this dataset also contains protected
                                                         # attribute for "sex" which we do not
                                                         # consider in this evaluation
            privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
            features_to_drop=['personal_status', 'sex']  # ignore sex-related attributes
        )
        
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_train = RW.fit_transform(dataset_orig_train)
        
        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        display(Markdown("#### Transformed training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

        # import from the generated python code 
        import tutorial_credit_scoring
        summary = tutorial_credit_scoring.fairml.bias_mitigations[0].summary_table
        #----------------------------------------
         
        self.assertAlmostEqual(summary.at[1, 'mean_difference'],
                               metric_orig_train.mean_difference(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'mean_difference'],
                               metric_transf_train.mean_difference(), delta=self.tolerance)
        print("")
    
    def test_demo_short_gerryfair_test(self):
        
        # import from the generated python code 
        import demo_short_gerryfair_test
        summary = demo_short_gerryfair_test.fairml.bias_mitigations[0].summary_table
        #----------------------------------------
        
        import warnings
        warnings.filterwarnings("ignore")
        # import sys
        # sys.path.append("../")
        from aif360.algorithms.inprocessing import GerryFairClassifier
        from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        from sklearn import svm
        from sklearn import tree
        from sklearn.kernel_ridge import KernelRidge
        from sklearn import linear_model
        from aif360.metrics import BinaryLabelDatasetMetric
        from IPython.display import Image
        import pickle
        import matplotlib.pyplot as plt
        import numpy as np
        np.random.seed(0)
        
        # load data set
        data_set = load_preproc_data_adult(sub_samp=1000, balance=True)
        max_iterations = 10
        C = 100
        print_flag = True
        gamma = .005
        
        fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                     max_iters=max_iterations, heatmapflag=False)
        
        fair_model.fit(data_set, early_termination=True)
        
        dataset_yhat = fair_model.predict(data_set, threshold=False)
        
        gerry_metric = BinaryLabelDatasetMetric(data_set)
        gamma_disparity = gerry_metric.rich_subgroup(array_to_tuple(dataset_yhat.labels), 'FP')
        print(gamma_disparity)
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[1, 'rich_subgroup'],
                               gamma_disparity, delta=self.tolerance)
        
        pareto_iters = 10

        # def multiple_classifiers_pareto(dataset, gamma_list=[0.002, 0.005, 0.01], save_results=False, iters=pareto_iters):
        def multiple_classifiers_pareto(dataset, gamma_list=[1.0], save_results=False, iters=pareto_iters):
        
            ln_predictor = linear_model.LinearRegression()
            svm_predictor = svm.LinearSVR()
            tree_predictor = tree.DecisionTreeRegressor(max_depth=3)
            kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
            predictor_dict = {'Linear': {'predictor': ln_predictor, 'iters': iters},
                              'SVR': {'predictor': svm_predictor, 'iters': iters},
                              'Tree': {'predictor': tree_predictor, 'iters': iters},
                              'Kernel': {'predictor': kernel_predictor, 'iters': iters}}
        
            results_dict = {}
        
            pos = 2
            for pred in predictor_dict:
                print('Curr Predictor: {}'.format(pred))
                predictor = predictor_dict[pred]['predictor']
                max_iters = predictor_dict[pred]['iters']
                fair_clf = GerryFairClassifier(C=100, printflag=True, gamma=1, predictor=predictor, max_iters=max_iters)
                fair_clf.printflag = False
                fair_clf.max_iters = max_iters
                errors, fp_violations, fn_violations = fair_clf.pareto(dataset, gamma_list)
                results_dict[pred] = {'errors': errors, 'fp_violations': fp_violations, 'fn_violations': fn_violations}
                
                ''' ASSERT '''
                print('Actual: {}, Expected: {}'.format(summary.at[pos, 'rich_subgroup'], fp_violations[0]))
                self.assertAlmostEqual(summary.at[pos, 'rich_subgroup'],
                               fp_violations[0], delta=self.tolerance)
                pos = pos + 1
                
            if save_results:
                pickle.dump(results_dict, open('results_dict_' + str(gamma_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))
        
        multiple_classifiers_pareto(data_set)
        print("")

    def test_demo_optimized_preprocessing_adult(self):
        
        # import from the generated python code 
        import demo_optim_preproc_adult
        summary = demo_optim_preproc_adult.fairml.bias_mitigations[0].summary_table
        #----------------------------------------
        
        import numpy as np
        from aif360.metrics import BinaryLabelDatasetMetric
        
        from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions            import load_preproc_data_adult
        from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions            import get_distortion_adult
        from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
        
        from IPython.display import Markdown, display
        
        np.random.seed(0)
        
        dataset_orig = load_preproc_data_adult(['race'])
        
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        
        privileged_groups = [{'race': 1}]  # White
        unprivileged_groups = [{'race': 0}]  # Not white
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[1, 'mean_difference'],
                               metric_orig_train.mean_difference(), delta=self.tolerance)
        
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
        
        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        display(Markdown("#### Transformed training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[2, 'mean_difference'],
                               metric_transf_train.mean_difference(), delta=self.tolerance)
    
    def test_demo_meta_classifier(self):
        '''
            Test Demo Meta Classifier
        '''
        
        # import from the generated python code 
        import demo_meta_classifier
        summary1 = demo_meta_classifier.fairml.bias_mitigations[0].summary_table
        summary2 = demo_meta_classifier.fairml.bias_mitigations[1].summary_table
        #----------------------------------------
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.preprocessing import MaxAbsScaler
        from tqdm import tqdm
        
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.metrics import ClassificationMetric
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        from aif360.algorithms.inprocessing import MetaFairClassifier
        
        np.random.seed(0)
        
        dataset_orig = load_preproc_data_adult()
        
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        
        min_max_scaler = MaxAbsScaler()
        dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
        
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        print("Train set: Difference in mean outcomes between unprivileged and privileged groups = {:.6f}".format(metric_orig_train.mean_difference()))
    
        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {:.6f}".format(metric_orig_test.mean_difference()))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[1, 'mean_difference'], metric_orig_test.mean_difference(), delta=self.tolerance)
        ''' ------ '''
        
        biased_model = MetaFairClassifier(tau=0, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)
        
        dataset_bias_test = biased_model.predict(dataset_orig_test)
        
        classified_metric_bias_test = ClassificationMetric(dataset_orig_test, dataset_bias_test,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = {:.6f}".format(classified_metric_bias_test.accuracy()))
        TPR = classified_metric_bias_test.true_positive_rate()
        TNR = classified_metric_bias_test.true_negative_rate()
        bal_acc_bias_test = 0.5 * (TPR + TNR)
        print("Test set: Balanced classification accuracy = {:.6f}".format(bal_acc_bias_test))
        print("Test set: Disparate impact = {:.6f}".format(classified_metric_bias_test.disparate_impact()))
        fdr = classified_metric_bias_test.false_discovery_rate_ratio()
        # fdr = min(fdr, 1/fdr)
        print("Test set: False discovery rate ratio = {:.6f}".format(fdr))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[2, 'accuracy'], classified_metric_bias_test.accuracy(), delta=self.tolerance)
        self.assertAlmostEqual(summary1.at[2, 'balanced_accuracy'], bal_acc_bias_test, delta=self.tolerance)
        self.assertAlmostEqual(summary1.at[2, 'disparate_impact'], classified_metric_bias_test.disparate_impact(), delta=self.tolerance)
        self.assertAlmostEqual(summary1.at[2, 'false_discovery_rate_ratio'], classified_metric_bias_test.false_discovery_rate_ratio(), delta=self.tolerance)
        ''' ------ '''
        
        debiased_model = MetaFairClassifier(tau=0.7, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)
        
        dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
        
        metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {:.6f}".format(metric_dataset_debiasing_test.mean_difference()))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[3, 'mean_difference'], metric_dataset_debiasing_test.mean_difference(), delta=self.tolerance)
        ''' ------ '''
        
        classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_debiasing_test,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = {:.6f}".format(classified_metric_debiasing_test.accuracy()))
        TPR = classified_metric_debiasing_test.true_positive_rate()
        TNR = classified_metric_debiasing_test.true_negative_rate()
        bal_acc_debiasing_test = 0.5 * (TPR + TNR)
        print("Test set: Balanced classification accuracy = {:.6f}".format(bal_acc_debiasing_test))
        print("Test set: Disparate impact = {:.6f}".format(classified_metric_debiasing_test.disparate_impact()))
        fdr = classified_metric_debiasing_test.false_discovery_rate_ratio()
        # fdr = min(fdr, 1/fdr)
        print("Test set: False discovery rate ratio = {:.6f}".format(fdr))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[3, 'accuracy'], classified_metric_debiasing_test.accuracy(), delta=self.tolerance)
        self.assertAlmostEqual(summary1.at[3, 'balanced_accuracy'], bal_acc_debiasing_test, delta=self.tolerance)
        self.assertAlmostEqual(summary1.at[3, 'disparate_impact'], classified_metric_debiasing_test.disparate_impact(), delta=self.tolerance)
        self.assertAlmostEqual(summary1.at[3, 'false_discovery_rate_ratio'], classified_metric_debiasing_test.false_discovery_rate_ratio(), delta=self.tolerance)
        ''' ------ '''
        
        # dataset_orig = load_preproc_data_adult()
        # privileged_groups = [{'sex': 1}]
        # unprivileged_groups = [{'sex': 0}]
        # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        # min_max_scaler = MaxAbsScaler()
        # dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
        # dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
        # np.random.seed(0)
        accuracies, disparates, statistical_rates = [], [], []
        s_attr = 'race'
        
        all_tau = np.linspace(0, 0.9, 10)
        for tau in tqdm(all_tau):
            debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=s_attr, type='sr')
            debiased_model = debiased_model.fit(dataset_orig_train)
        
            dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
            metric = ClassificationMetric(dataset_orig_test, dataset_debiasing_test,
                                          unprivileged_groups=[{s_attr: 0}],
                                          privileged_groups=[{s_attr: 1}])
        
            accuracies.append(metric.accuracy())
            sr = metric.disparate_impact()
            disparates.append(sr)
            statistical_rates.append(1 - abs(sr - 1))
        #     statistical_rates.append(min(sr, 1/sr))
        
        for i in range(len(accuracies)):
            print("Tau: {:.1f}, Accuracy: {:.6f}, Disparate Impact: {:.6f}, Rates: {:.6}".format(all_tau[i], accuracies[i], disparates[i], statistical_rates[i]))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary2.at[1, 'accuracy'], accuracies[0], delta=self.tolerance)
        self.assertAlmostEqual(summary2.at[1, 'disparate_impact'], disparates[0], delta=self.tolerance)
        
        gen_accuracies = summary2["accuracy"].values.tolist()[1:11]
        gen_impacts = summary2["disparate_impact"].values.tolist()[1:11]
        avg_gen_accuracies = sum(gen_accuracies) / len(gen_accuracies) 
        avg_gen_impacts = sum(gen_impacts) / len(gen_impacts) 
        avg_demo_accuracies = sum(accuracies) / len(accuracies) 
        avg_demo_impacts = sum(disparates) / len(disparates) 
        
        self.assertAlmostEqual(avg_gen_accuracies, avg_demo_accuracies, delta=0.1)
        self.assertAlmostEqual(avg_gen_impacts, avg_demo_impacts, delta=0.1)
        
        ''' ------ '''
        
        fig, ax1 = plt.subplots(figsize=(13, 7))
        ax1.plot(all_tau, accuracies, color='b')
        ax1.set_title('Accuracy and $\gamma_{sr}$ vs Tau', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Input Tau', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        ax2 = ax1.twinx()
        ax2.plot(all_tau, statistical_rates, color='r')
        ax2.set_ylabel('$\gamma_{sr}$', color='r', fontsize=16, fontweight='bold')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
    
    def test_demo_disparate_impact_remover(self):
        '''
            Test Against Demo Disparate Impact Remover
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_disparate_impact_remover.ipynb
        '''
        
        # import from the generated python code 
        import demo_disparate_impact_remover
        summary = demo_disparate_impact_remover.fairml.bias_mitigations[0].summary_table
        #----------------------------------------
        
        # from __future__ import absolute_import
        # from __future__ import division
        # from __future__ import print_function
        # from __future__ import unicode_literals
        
        from matplotlib import pyplot as plt
        
        import sys
        sys.path.append("../")
        import warnings
        
        import numpy as np
        np.random.seed(0)
        from tqdm import tqdm
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC as SVM
        from sklearn.preprocessing import MinMaxScaler
        
        from aif360.algorithms.preprocessing import DisparateImpactRemover
        from aif360.datasets import AdultDataset
        from aif360.metrics import BinaryLabelDatasetMetric

        protected = 'sex'
        ad = AdultDataset(protected_attribute_names=[protected],
            privileged_classes=[['Male']], categorical_features=[],
            features_to_drop=['fnlwgt'],
            features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])
        
        scaler = MinMaxScaler(copy=False)
        
        test, train = ad.split([16281])
        train.features = scaler.fit_transform(train.features)
        test.features = scaler.fit_transform(test.features)
        
        index = train.feature_names.index(protected)
        
        DIs = []
        for level in tqdm(np.linspace(0., 1., 11)):
            di = DisparateImpactRemover(repair_level=level)
            train_repd = di.fit_transform(train)
            test_repd = di.fit_transform(test)
            
            X_tr = np.delete(train_repd.features, index, axis=1)
            X_te = np.delete(test_repd.features, index, axis=1)

            y_tr = train_repd.labels.ravel()
            
            lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
            lmod.fit(X_tr, y_tr)
            
            test_repd_pred = test_repd.copy()
            test_repd_pred.labels = lmod.predict(X_te)
        
            p = [{protected: 1}]
            u = [{protected: 0}]
            cm = BinaryLabelDatasetMetric(test_repd_pred, privileged_groups=p, unprivileged_groups=u)
            DIs.append(cm.disparate_impact())
        
        x = 0 
        for val in DIs:
            print(str(x) + ": " + str(val))
            
            ''' ASSERT '''
            self.assertAlmostEqual(summary.at[x + 2, 'disparate_impact'], val, delta=self.tolerance)
            ''' ------ '''

            x = x + 1
        
        plt.plot(np.linspace(0, 1, 11), DIs, marker='o')
        plt.plot([0, 1], [1, 1], 'g')
        plt.plot([0, 1], [0.8, 0.8], 'r')
        plt.ylim([0.0, 1.2])
        plt.ylabel('Disparate Impact (DI)')
        plt.xlabel('repair level')
        plt.show()

    def test_demo_exponentiated_gradient_reduction(self):
        '''
            Test Against Demo Exponentiated Gradient Reduction
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_exponentiated_gradient_reduction.ipynb
        '''
        
        # import from the generated python code 
        import demo_exponentiated_gradient_reduction
        summary = demo_exponentiated_gradient_reduction.fairml.bias_mitigations[0].summary_table
        #----------------------------------------
        
        # import sys
        # sys.path.append("../")
        from aif360.datasets import BinaryLabelDataset
        from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.metrics import ClassificationMetric
        
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
        
        from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, MaxAbsScaler
        from sklearn.metrics import accuracy_score
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        
        import numpy as np
        np.random.seed(0)
        
        dataset_orig = load_preproc_data_adult()
        
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
        
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)
    
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
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
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[1, 'mean_difference'], metric_scaled_test.mean_difference(), delta=self.tolerance)
        ''' ------ '''
        
        X_train = dataset_orig_train.features
        y_train = dataset_orig_train.labels.ravel()
        
        lmod = LogisticRegression(solver='lbfgs')
        lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)
        
        X_test = dataset_orig_test.features
        y_test = dataset_orig_test.labels.ravel()
        
        y_pred = lmod.predict(X_test)
        
        display(Markdown("#### Accuracy"))
        lr_acc = accuracy_score(y_test, y_pred)
        print(lr_acc)
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[2, 'accuracy'], lr_acc, delta=self.tolerance)
        ''' ------ '''
        
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.labels = y_pred
        
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
        
        metric_test = ClassificationMetric(dataset_orig_test,
                                            dataset_orig_test_pred,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
        display(Markdown("#### Average odds difference"))
        lr_aod = metric_test.average_odds_difference()
        print(lr_aod)
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[2, 'average_odds_difference'], lr_aod, delta=self.tolerance)
        ''' ------ '''
        
        np.random.seed(0)  # need for reproducibility
        estimator = LogisticRegression(solver='lbfgs')
    
        exp_grad_red = ExponentiatedGradientReduction(estimator=estimator,
                                                      constraints="EqualizedOdds",
                                                      drop_prot_attr=False)
        exp_grad_red.fit(dataset_orig_train)
        exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)
        
        metric_test = ClassificationMetric(dataset_orig_test,
                                           exp_grad_red_pred,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
        
        display(Markdown("#### Accuracy"))
        egr_acc = metric_test.accuracy()
        print(egr_acc)
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[3, 'accuracy'], egr_acc, delta=self.tolerance)
        self.assertLess(abs(summary.at[2, 'accuracy'] - summary.at[3, 'accuracy']), 0.03)
        ''' ------ '''
        
        assert abs(lr_acc - egr_acc) < 0.03
        
        display(Markdown("#### Average odds difference"))
        egr_aod = metric_test.average_odds_difference()
        print(egr_aod)
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[3, 'average_odds_difference'], egr_aod, delta=self.tolerance)
        self.assertLess(summary.at[3, 'average_odds_difference'], abs(summary.at[2, 'average_odds_difference']))
        ''' ------ '''
        
        assert abs(egr_aod) < abs(lr_aod)

    def test_demo_reject_option_classification(self):
        ''''
            Reject Option Classification (ROC) post-processing algorithm for bias mitigation
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_reject_option_classification.ipynb
        '''
        
        # import from the generated python code 
        import demo_reject_option_classification
        summary = demo_reject_option_classification.fairml.bias_mitigations[0].summary_table
        #--------------------------------
        
        import sys
        import numpy as np
        from tqdm import tqdm
        from warnings import warn
        
        from aif360.datasets import BinaryLabelDataset
        from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
        from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
        from aif360.metrics.utils import compute_boolean_conditioning_vector
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
        from aif360.algorithms.postprocessing.reject_option_classification        import RejectOptionClassification
        from common_utils import compute_metrics
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        from ipywidgets import interactive, FloatSlider
        
        # #### Load dataset and specify options
        
        # In[2]:
        
        # # import dataset
        dataset_used = "adult"  # "adult", "german", "compas"
        protected_attribute_used = 1  # 1, 2
        
        if dataset_used == "adult":
        #     dataset_orig = AdultDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
                dataset_orig = load_preproc_data_adult(['sex'])
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
                dataset_orig = load_preproc_data_adult(['race'])
            
        elif dataset_used == "german":
        #     dataset_orig = GermanDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
                dataset_orig = load_preproc_data_german(['sex'])
            else:
                privileged_groups = [{'age': 1}]
                unprivileged_groups = [{'age': 0}]
                dataset_orig = load_preproc_data_german(['age'])
            
        elif dataset_used == "compas":
        #     dataset_orig = CompasDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
                dataset_orig = load_preproc_data_compas(['sex'])
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]  
                dataset_orig = load_preproc_data_compas(['race'])
                
        # Metric used (should be one of allowed_metrics)
        metric_name = "Statistical parity difference"
        
        # Upper and lower bound on the fairness metric used
        metric_ub = 0.05
        metric_lb = -0.05
                
        np.random.seed(0)
        
        allowed_metrics = ["Statistical parity difference",
                           "Average odds difference",
                           "Equal opportunity difference"]
        if metric_name not in allowed_metrics:
            raise ValueError("Metric name should be one of allowed metrics")
        
        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
        
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)

        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

        '''ASSERT'''
        self.assertAlmostEqual(summary.at[1, 'mean_difference'],
                               metric_orig_train.mean_difference(), delta=self.tolerance)
        '''------'''
        
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)
        y_train_pred = lmod.predict(X_train)
        
        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_train_pred.labels = y_train_pred
        
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
        y_valid = dataset_orig_valid_pred.labels
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)
        
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = scale_orig.transform(dataset_orig_test_pred.features)
        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
        
        num_thresh = 100
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, 1.00, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):
            
            fav_inds = dataset_orig_valid_pred.scores > class_thresh
            dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
            dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
            
            classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                     dataset_orig_valid_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
            
            ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate() + classified_metric_orig_valid.true_negative_rate())
        
        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_class_thresh = class_thresh_arr[best_ind]
        
        print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
        print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)
        
        ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups,
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                          num_class_thresh=100, num_ROC_margin=50,
                                          metric_name=metric_name,
                                          metric_ub=metric_ub, metric_lb=metric_lb)
        ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
        
        print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
        print("Optimal ROC margin = %.4f" % ROC.ROC_margin)
        
        fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        display(Markdown("#### Validation set"))
        display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))
        
        metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred,
                        unprivileged_groups, privileged_groups)
        
        '''ASSERT'''
        self.assertAlmostEqual(summary.at[2, 'balanced_accuracy'],
                               metric_valid_bef['Balanced accuracy'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'statistical_parity_difference'],
                               metric_valid_bef['Statistical parity difference'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'disparate_impact'],
                               metric_valid_bef['Disparate impact'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'average_odds_difference'],
                               metric_valid_bef['Average odds difference'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'equal_opportunity_difference'],
                               metric_valid_bef['Equal opportunity difference'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'theil_index'],
                               metric_valid_bef['Theil index'], delta=self.tolerance)
        '''------'''
        
        dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)
        
        display(Markdown("#### Validation set"))
        display(Markdown("##### Transformed predictions - With fairness constraints"))
        metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred,
                        unprivileged_groups, privileged_groups)
        
        print(summary.iloc[0])
        print(summary.iloc[1])
        print(summary.iloc[2])
        print(metric_valid_aft)
        
        '''ASSERT'''
        self.assertAlmostEqual(summary.at[3, 'balanced_accuracy'],
                               metric_valid_aft['Balanced accuracy'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'statistical_parity_difference'],
                               metric_valid_aft['Statistical parity difference'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'disparate_impact'],
                               metric_valid_aft['Disparate impact'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'average_odds_difference'],
                               metric_valid_aft['Average odds difference'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'equal_opportunity_difference'],
                               metric_valid_aft['Equal opportunity difference'], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'theil_index'],
                               metric_valid_aft['Theil index'], delta=self.tolerance)
        '''------'''
        
        assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])
        
        fav_inds = dataset_orig_test_pred.scores > best_class_thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
        
        display(Markdown("#### Test set"))
        display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))
        
        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                        unprivileged_groups, privileged_groups)
        
        dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
        
        display(Markdown("#### Test set"))
        display(Markdown("##### Transformed predictions - With fairness constraints"))
        metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                        unprivileged_groups, privileged_groups)

    def test_demo_adversarial_debiasing(self):
        ''''
            Demo Adversarial Debiasing
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_adversarial_debiasing.ipynb
        '''
        
        # import from the generated python code 
        import demo_adversarial_debiasing
        summary = demo_adversarial_debiasing.fairml.bias_mitigations[0].summary_table
        #--------------------------------
        
        import sys
        import numpy as np
        np.random.seed(0)
        from aif360.datasets import BinaryLabelDataset
        from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.metrics import ClassificationMetric
        from aif360.metrics.utils import compute_boolean_conditioning_vector
        
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
        
        from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, MaxAbsScaler
        from sklearn.metrics import accuracy_score
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
    
        dataset_orig = load_preproc_data_adult()
        
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)

        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
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
        
        sess = tf.Session()
        plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups,
                                  scope_name='plain_classifier',
                                  debias=False,
                                  sess=sess)
        
        plain_model.fit(dataset_orig_train)
        
        dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
        dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)
        
        display(Markdown("#### Plain model - without debiasing - dataset metrics"))
        metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        
        print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
        
        metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())
        
        display(Markdown("#### Plain model - without debiasing - classification metrics"))
        classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_nodebiasing_test,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
        TPR = classified_metric_nodebiasing_test.true_positive_rate()
        TNR = classified_metric_nodebiasing_test.true_negative_rate()
        bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
        print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
        print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
        print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
        print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
        print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
        
        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()
        
        debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups,
                                  scope_name='debiased_classifier',
                                  debias=True,
                                  sess=sess)
        
        debiased_model.fit(dataset_orig_train)
        
        dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
        dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
        
        display(Markdown("#### Plain model - without debiasing - dataset metrics"))
        print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())
        
        display(Markdown("#### Model - with debiasing - dataset metrics"))
        metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        
        print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())
        
        self.assertAlmostEqual(summary.at[1, 'mean_difference'], metric_dataset_debiasing_train.mean_difference(), delta=self.tolerance)
        
        metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())
        
        display(Markdown("#### Plain model - without debiasing - classification metrics"))
        print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
        TPR = classified_metric_nodebiasing_test.true_positive_rate()
        TNR = classified_metric_nodebiasing_test.true_negative_rate()
        bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
        print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
        print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
        print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
        print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
        print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
        
        self.assertAlmostEqual(summary.at[2, 'accuracy'], classified_metric_nodebiasing_test.accuracy(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'balanced_accuracy'], bal_acc_nodebiasing_test, delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'disparate_impact'], classified_metric_nodebiasing_test.disparate_impact(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'equal_opportunity_difference'], classified_metric_nodebiasing_test.equal_opportunity_difference(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'average_odds_difference'], classified_metric_nodebiasing_test.average_odds_difference(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'theil_index'], classified_metric_nodebiasing_test.theil_index(), delta=self.tolerance)
        
        display(Markdown("#### Model - with debiasing - classification metrics"))
        classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_debiasing_test,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
        TPR = classified_metric_debiasing_test.true_positive_rate()
        TNR = classified_metric_debiasing_test.true_negative_rate()
        bal_acc_debiasing_test = 0.5 * (TPR + TNR)
        print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
        print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
        print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
        print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
        print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())
        
        self.assertAlmostEqual(summary.at[3, 'accuracy'], classified_metric_debiasing_test.accuracy(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'balanced_accuracy'], bal_acc_debiasing_test, delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'disparate_impact'], classified_metric_debiasing_test.disparate_impact(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'equal_opportunity_difference'], classified_metric_debiasing_test.equal_opportunity_difference(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'average_odds_difference'], classified_metric_debiasing_test.average_odds_difference(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'theil_index'], classified_metric_debiasing_test.theil_index(), delta=self.tolerance)

    def test_demo_lfr(self):
        ''''
            Demo Learning Fair Representations
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_lfr.ipynb
        '''
        
        # import from the generated python code 
        import demo_lfr
        summary = demo_lfr.fairml.bias_mitigations[0].summary_table
        #--------------------------------
        
        import sys
        sys.path.append("../")
        import numpy as np
        np.random.seed(0)
        from aif360.datasets import BinaryLabelDataset
        from aif360.datasets import AdultDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.metrics import ClassificationMetric
        from aif360.metrics.utils import compute_boolean_conditioning_vector
        
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        from aif360.algorithms.preprocessing.lfr import LFR
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        import numpy as np
        
        from common_utils import compute_metrics
    
        dataset_orig = load_preproc_data_adult()
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)
       
        privileged_groups = [{'sex': 1.0}]
        unprivileged_groups = [{'sex': 0.0}]
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        
        self.assertAlmostEqual(summary.at[1, 'mean_difference'], metric_orig_train.mean_difference(), delta=self.tolerance)
        
        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original test dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
        
        scale_orig = StandardScaler()
        dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = scale_orig.transform(dataset_orig_test.features)
     
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
            
        TR = LFR(unprivileged_groups=unprivileged_groups,
                 privileged_groups=privileged_groups,
                 k=10, Ax=0.1, Ay=1.0, Az=2.0,
                 verbose=1
                )
        TR = TR.fit(dataset_orig_train, maxiter=5000, maxfun=5000)
        
        dataset_transf_train = TR.transform(dataset_orig_train)
        dataset_transf_test = TR.transform(dataset_orig_test)

        print(classification_report(dataset_orig_test.labels, dataset_transf_test.labels))

        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Transformed training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
        metric_transf_test = BinaryLabelDatasetMetric(dataset_transf_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Transformed test dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_test.mean_difference())

        from common_utils import compute_metrics
        
        display(Markdown("#### Predictions from transformed testing data"))
        bal_acc_arr_transf = []
        disp_imp_arr_transf = []
        
        class_thresh_arr = np.linspace(0.01, 1.00, 100)
        
        dataset_transf_test_new = dataset_orig_test.copy(deepcopy=True)
        dataset_transf_test_new.scores = dataset_transf_test.scores
        
        for thresh in class_thresh_arr:
            
            fav_inds = dataset_transf_test_new.scores > thresh
            dataset_transf_test_new.labels[fav_inds] = 1.0
            dataset_transf_test_new.labels[~fav_inds] = 0.0
            
            metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_new,
                                              unprivileged_groups, privileged_groups,
                                              disp=False)
        
            bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
            disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])
        
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax1.plot(class_thresh_arr, bal_acc_arr_transf)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        ax2 = ax1.twinx()
        ax2.plot(class_thresh_arr, np.abs(1.0 - np.array(disp_imp_arr_transf)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        
        display(Markdown("#### Individual fairness metrics"))
        print("Consistency of labels in transformed training dataset= %f" % metric_transf_train.consistency())
        print("Consistency of labels in original training dataset= %f" % metric_orig_train.consistency())
        print("Consistency of labels in transformed test dataset= %f" % metric_transf_test.consistency())
        print("Consistency of labels in original test dataset= %f" % metric_orig_test.consistency())
        
        self.assertAlmostEqual(summary.at[1, 'consistency'], metric_orig_train.consistency(), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'consistency'], metric_transf_train.consistency(), delta=self.tolerance)
        
        def check_algorithm_success():
            """Transformed dataset consistency should be greater than original dataset."""
            assert metric_transf_test.consistency() > metric_orig_test.consistency(), "Transformed dataset consistency should be greater than original dataset."
        
        check_algorithm_success()    


    def test_demo_calibrated_eqodds_postprocessing(self):
        ''''
            Demo Calibrated Eqodds Postprocessing
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_calibrated_eqodds_postprocessing.ipynb
        '''
        
        # import from the generated python code 
        import demo_calibrated_eqodds_postprocessing
        summary = demo_calibrated_eqodds_postprocessing.fairml.bias_mitigations[0].summary_table
        #--------------------------------

        import sys
        import numpy as np
        np.random.seed(0)
        import pandas as pd
        
        sys.path.append("../")
        from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.metrics import ClassificationMetric
        from aif360.metrics.utils import compute_boolean_conditioning_vector
        
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas
        
        from sklearn.preprocessing import scale
        from sklearn.linear_model import LogisticRegression
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        
        dataset_used = "adult"  # "adult", "german", "compas"
        protected_attribute_used = 1  # 1, 2
        
        if dataset_used == "adult":
            dataset_orig = AdultDataset()
        #     dataset_orig = load_preproc_data_adult()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
            
        elif dataset_used == "german":
            dataset_orig = GermanDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'age': 1}]
                unprivileged_groups = [{'age': 0}]
            
        elif dataset_used == "compas":
        #     dataset_orig = CompasDataset()
            dataset_orig = load_preproc_data_compas()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]    
        
        cost_constraint = "fnr"  # "fnr", "fpr", "weighted"
        randseed = 0 
        
        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
        
        display(Markdown("#### Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        
        self.assertAlmostEqual(summary.at[1, 'mean_difference'], metric_orig_train.mean_difference(), delta=self.tolerance)
        
        metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original validation dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())
        
        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original test dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
    
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_curve
        
        # Placeholder for predicted and transformed datasets
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        
        dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)
        
        # Logistic regression classifier and predictions for training data
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)
        
        fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]
        
        # Prediction probs for validation and testing data
        X_valid = scale_orig.transform(dataset_orig_valid.features)
        y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]
        
        X_test = scale_orig.transform(dataset_orig_test.features)
        y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]
        
        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
        dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)
        
        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred
        
        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred
            
        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred
        
        cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original-Predicted training dataset"))
        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_pred_train.difference(cm_pred_train.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_pred_train.difference(cm_pred_train.generalized_false_negative_rate))
        
        cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original-Predicted validation dataset"))
        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_pred_valid.difference(cm_pred_valid.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))
        
        cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original-Predicted testing dataset"))
        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate))
        
        self.assertAlmostEqual(summary.at[2, 'generalized_false_positive_rate_difference'], cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate), delta=self.tolerance)
        # self.assertAlmostEqual(summary.at[2, 'generalized_false_negative_rate_difference'], cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate), delta=self.tolerance)
        
        from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
        from tqdm import tqdm
        
        cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                             unprivileged_groups=unprivileged_groups,
                                             cost_constraint=cost_constraint,
                                             seed=randseed)
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
        
        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
        
        cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original-Transformed validation dataset"))
        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_transf_valid.difference(cm_transf_valid.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate))
        
        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original-Transformed testing dataset"))
        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate))
        
        self.assertAlmostEqual(summary.at[3, 'generalized_false_positive_rate_difference'], cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate), delta=self.tolerance)
        self.assertAlmostEqual(summary.at[3, 'generalized_false_negative_rate_difference'], cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate), delta=self.tolerance)
        
        assert np.abs(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate)) < np.abs(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate)) 

        all_thresh = np.linspace(0.01, 1.00, 25)
        display(Markdown("#### Classification thresholds used for validation and parameter selection"))
        
        bef_avg_odds_diff_test = []
        bef_avg_odds_diff_valid = []
        aft_avg_odds_diff_test = []
        aft_avg_odds_diff_valid = []
        bef_bal_acc_valid = []
        bef_bal_acc_test = []
        aft_bal_acc_valid = []
        aft_bal_acc_test = []
        for thresh in tqdm(all_thresh):
            
            dataset_orig_valid_pred_thresh = dataset_orig_valid_pred.copy(deepcopy=True)
            dataset_orig_test_pred_thresh = dataset_orig_test_pred.copy(deepcopy=True)
            dataset_transf_valid_pred_thresh = dataset_transf_valid_pred.copy(deepcopy=True)
            dataset_transf_test_pred_thresh = dataset_transf_test_pred.copy(deepcopy=True)
            
            # Labels for the datasets from scores
            y_temp = np.zeros_like(dataset_orig_valid_pred_thresh.labels)
            y_temp[dataset_orig_valid_pred_thresh.scores >= thresh] = dataset_orig_valid_pred_thresh.favorable_label
            y_temp[~(dataset_orig_valid_pred_thresh.scores >= thresh)] = dataset_orig_valid_pred_thresh.unfavorable_label
            dataset_orig_valid_pred_thresh.labels = y_temp
        
            y_temp = np.zeros_like(dataset_orig_test_pred_thresh.labels)
            y_temp[dataset_orig_test_pred_thresh.scores >= thresh] = dataset_orig_test_pred_thresh.favorable_label
            y_temp[~(dataset_orig_test_pred_thresh.scores >= thresh)] = dataset_orig_test_pred_thresh.unfavorable_label
            dataset_orig_test_pred_thresh.labels = y_temp
            
            y_temp = np.zeros_like(dataset_transf_valid_pred_thresh.labels)
            y_temp[dataset_transf_valid_pred_thresh.scores >= thresh] = dataset_transf_valid_pred_thresh.favorable_label
            y_temp[~(dataset_transf_valid_pred_thresh.scores >= thresh)] = dataset_transf_valid_pred_thresh.unfavorable_label
            dataset_transf_valid_pred_thresh.labels = y_temp
            
            y_temp = np.zeros_like(dataset_transf_test_pred_thresh.labels)
            y_temp[dataset_transf_test_pred_thresh.scores >= thresh] = dataset_transf_test_pred_thresh.favorable_label
            y_temp[~(dataset_transf_test_pred_thresh.scores >= thresh)] = dataset_transf_test_pred_thresh.unfavorable_label
            dataset_transf_test_pred_thresh.labels = y_temp
            
            # Metrics for original validation data
            classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                         dataset_orig_valid_pred_thresh,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
            bef_avg_odds_diff_valid.append(classified_metric_orig_valid.equal_opportunity_difference())
        
            bef_bal_acc_valid.append(0.5 * (classified_metric_orig_valid.true_positive_rate() + 
                                      classified_metric_orig_valid.true_negative_rate()))
        
            classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_orig_test_pred_thresh,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
            bef_avg_odds_diff_test.append(classified_metric_orig_test.equal_opportunity_difference())
            bef_bal_acc_test.append(0.5 * (classified_metric_orig_test.true_positive_rate() + 
                                      classified_metric_orig_test.true_negative_rate()))
        
            # Metrics for transf validing data
            classified_metric_transf_valid = ClassificationMetric(
                                             dataset_orig_valid,
                                             dataset_transf_valid_pred_thresh,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
            aft_avg_odds_diff_valid.append(classified_metric_transf_valid.equal_opportunity_difference())
            aft_bal_acc_valid.append(0.5 * (classified_metric_transf_valid.true_positive_rate() + 
                                      classified_metric_transf_valid.true_negative_rate()))
        
            # Metrics for transf validation data
            classified_metric_transf_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_transf_test_pred_thresh,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
            aft_avg_odds_diff_test.append(classified_metric_transf_test.equal_opportunity_difference())
            aft_bal_acc_test.append(0.5 * (classified_metric_transf_test.true_positive_rate() + 
                                          classified_metric_transf_test.true_negative_rate()))
        
        bef_bal_acc_valid = np.array(bef_bal_acc_valid)
        bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)
        
        aft_bal_acc_valid = np.array(aft_bal_acc_valid)
        aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)
        
        fig, ax1 = plt.subplots(figsize=(13, 7))
        ax1.plot(all_thresh, bef_bal_acc_valid, color='b')
        ax1.plot(all_thresh, aft_bal_acc_valid, color='b', linestyle='dashed')
        ax1.set_title('Original and Postprocessed validation data', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        ax2 = ax1.twinx()
        ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_valid), color='r')
        ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_valid), color='r', linestyle='dashed')
        ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
                     "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc.", ],
                   fontsize=16)
        
        bef_bal_acc_test = np.array(bef_bal_acc_test)
        bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)
        
        aft_bal_acc_test = np.array(aft_bal_acc_test)
        aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)
        
        fig, ax1 = plt.subplots(figsize=(13, 7))
        ax1.plot(all_thresh, bef_bal_acc_test, color='b')
        ax1.plot(all_thresh, aft_bal_acc_test, color='b', linestyle='dashed')
        ax1.set_title('Original and Postprocessed testing data', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        ax2 = ax1.twinx()
        ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_test), color='r')
        ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_test), color='r', linestyle='dashed')
        ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
                    "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc."],
                   fontsize=16)
        

    def test_demo_reweighing_preproc(self):
        ''''
            This notebook demonstrates the use of a reweighing pre-processing algorithm for bias mitigation
            https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_reweighing_preproc.ipynb
        '''
        
        # import from the generated python code 
        import demo_reweighing_preproc
        summary = demo_reweighing_preproc.fairml.bias_mitigations[0].summary_table
        #--------------------------------
        
        import sys
        sys.path.append("../")
        import numpy as np
        np.random.seed(0)
        from tqdm import tqdm
        
        from aif360.datasets import BinaryLabelDataset
        from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.metrics import ClassificationMetric
        from aif360.algorithms.preprocessing.reweighing import Reweighing
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        
        from common_utils import compute_metrics
        
        dataset_used = "adult" # "adult", "german", "compas"
        protected_attribute_used = 1 # 1, 2
        
        
        if dataset_used == "adult":
        #     dataset_orig = AdultDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
                dataset_orig = load_preproc_data_adult(['sex'])
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
                dataset_orig = load_preproc_data_adult(['race'])
            
        elif dataset_used == "german":
        #     dataset_orig = GermanDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
                dataset_orig = load_preproc_data_german(['sex'])
            else:
                privileged_groups = [{'age': 1}]
                unprivileged_groups = [{'age': 0}]
                dataset_orig = load_preproc_data_german(['age'])
            
        elif dataset_used == "compas":
        #     dataset_orig = CompasDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
                dataset_orig = load_preproc_data_compas(['sex'])
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
                dataset_orig = load_preproc_data_compas(['race'])
        
        all_metrics =  ["Statistical parity difference",
                           "Average odds difference",
                           "Equal opportunity difference"]
        
        np.random.seed(0)
        
        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
        
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes, 
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)
        
    
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        
        self.assertAlmostEqual(summary.at[1, 'mean_difference'], metric_orig_train.mean_difference(), delta=self.tolerance)
        
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                       privileged_groups=privileged_groups)
        RW.fit(dataset_orig_train)
        dataset_transf_train = RW.transform(dataset_orig_train)
        
        assert np.abs(dataset_transf_train.instance_weights.sum()-dataset_orig_train.instance_weights.sum())<1e-6
        

        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        display(Markdown("#### Transformed training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
        
        assert np.abs(metric_transf_train.mean_difference()) < 1e-6
        
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        w_train = dataset_orig_train.instance_weights.ravel()
        
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train, 
                 sample_weight=dataset_orig_train.instance_weights)
        y_train_pred = lmod.predict(X_train)
        
        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        
        dataset_orig_train_pred = dataset_orig_train.copy()
        dataset_orig_train_pred.labels = y_train_pred
        
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
        y_valid = dataset_orig_valid_pred.labels
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)
        
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = scale_orig.transform(dataset_orig_test_pred.features)
        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
        
        num_thresh = 100
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, 1.00, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):
            
            fav_inds = dataset_orig_valid_pred.scores > class_thresh
            dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
            dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
            
            classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                     dataset_orig_valid_pred, 
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
            
            ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()                       +classified_metric_orig_valid.true_negative_rate())
        
        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_class_thresh = class_thresh_arr[best_ind]
        
        print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
        print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)
        
        
        display(Markdown("#### Predictions from original testing data"))
        bal_acc_arr_orig = []
        disp_imp_arr_orig = []
        avg_odds_diff_arr_orig = []
        
        print("Classification threshold used = %.4f" % best_class_thresh)
        for thresh in tqdm(class_thresh_arr):
            
            if thresh == best_class_thresh:
                disp = True
            else:
                disp = False
            
            fav_inds = dataset_orig_test_pred.scores > thresh
            dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
            dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
            
            metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                                              unprivileged_groups, privileged_groups,
                                              disp = disp)
            
            if disp == True:
                self.assertAlmostEqual(summary.at[2, 'balanced_accuracy'], metric_test_bef["Balanced accuracy"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[2, 'statistical_parity_difference'], metric_test_bef["Statistical parity difference"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[2, 'disparate_impact'], metric_test_bef["Disparate impact"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[2, 'average_odds_difference'], metric_test_bef["Average odds difference"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[2, 'equal_opportunity_difference'], metric_test_bef["Equal opportunity difference"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[2, 'theil_index'], metric_test_bef["Theil index"], delta=self.tolerance)
                    
            bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
            avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
            disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])
        
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(class_thresh_arr, bal_acc_arr_orig)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        ax2 = ax1.twinx()
        ax2.plot(class_thresh_arr, np.abs(1.0-np.array(disp_imp_arr_orig)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
        ax2.axvline(best_class_thresh, color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(class_thresh_arr, bal_acc_arr_orig)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        
        ax2 = ax1.twinx()
        ax2.plot(class_thresh_arr, avg_odds_diff_arr_orig, color='r')
        ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
        ax2.axvline(best_class_thresh, color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        
        scale_transf = StandardScaler()
        X_train = scale_transf.fit_transform(dataset_transf_train.features)
        y_train = dataset_transf_train.labels.ravel()
        
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train,
                sample_weight=dataset_transf_train.instance_weights)
        y_train_pred = lmod.predict(X_train)
        
        dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
        y_test = dataset_transf_test_pred.labels
        dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)
        
        display(Markdown("#### Predictions from transformed testing data"))
        bal_acc_arr_transf = []
        disp_imp_arr_transf = []
        avg_odds_diff_arr_transf = []
        
        print("Classification threshold used = %.4f" % best_class_thresh)
        for thresh in tqdm(class_thresh_arr):
            
            if thresh == best_class_thresh:
                disp = True
            else:
                disp = False
            
            fav_inds = dataset_transf_test_pred.scores > thresh
            dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
            dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label
            
            metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                                              unprivileged_groups, privileged_groups,
                                              disp = disp)
            
            if disp == True:
                self.assertAlmostEqual(summary.at[3, 'balanced_accuracy'], metric_test_aft["Balanced accuracy"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[3, 'statistical_parity_difference'], metric_test_aft["Statistical parity difference"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[3, 'disparate_impact'], metric_test_aft["Disparate impact"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[3, 'average_odds_difference'], metric_test_aft["Average odds difference"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[3, 'equal_opportunity_difference'], metric_test_aft["Equal opportunity difference"], delta=self.tolerance)
                self.assertAlmostEqual(summary.at[3, 'theil_index'], metric_test_aft["Theil index"], delta=self.tolerance)
        
            bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
            avg_odds_diff_arr_transf.append(metric_test_aft["Average odds difference"])
            disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])
        
        
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(class_thresh_arr, bal_acc_arr_transf)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        
        ax2 = ax1.twinx()
        ax2.plot(class_thresh_arr, np.abs(1.0-np.array(disp_imp_arr_transf)), color='r')
        ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
        ax2.axvline(best_class_thresh, color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        
        
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(class_thresh_arr, bal_acc_arr_transf)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        
        
        ax2 = ax1.twinx()
        ax2.plot(class_thresh_arr, avg_odds_diff_arr_transf, color='r')
        ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
        ax2.axvline(best_class_thresh, color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)


    def test_tutorial_medical_expenditure(self):
        ''''
            Medical Expenditure Tutorial.
            This tutorial demonstrates classification model learning with bias mitigation as a part of a Care Management use case using Medical Expenditure data.
            https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
        '''
        
        # import from the generated python code 
        import tutorial_medical_expenditure
        summary = tutorial_medical_expenditure.fairml.bias_mitigations[0].summary_table
        
        import sys
        sys.path.insert(0, '../')
        
        import matplotlib.pyplot as plt
        import numpy as np
        np.random.seed(0)
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
    
        
        (dataset_orig_panel19_train,
         dataset_orig_panel19_val,
         dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)
        
        sens_ind = 0
        sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]
        
        unprivileged_groups = [{sens_attr: v} for v in
                               dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
        privileged_groups = [{sens_attr: v} for v in
                             dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
        
        
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
        
        
        describe(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test)
    
        
        metric_orig_panel19_train = BinaryLabelDatasetMetric(
                dataset_orig_panel19_train,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
        explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)
        
        print(explainer_orig_panel19_train.disparate_impact())
        
        # ''' ASSERT '''
        # self.assertAlmostEqual(summary.at[1, 'disparate_impact'], explainer_orig_panel19_train.disparate_impact(), delta=self.tolerance)
        # ''' ------ '''
       
        dataset = dataset_orig_panel19_train
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(solver='liblinear', random_state=1))
        fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
        
        lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
        
    
        
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
            
            return metrics, best_ind
            
        
        metrics, best_ind = describe_metrics(val_metrics, thresh_arr)
        
        lr_orig_metrics = test(dataset=dataset_orig_panel19_test,
                               model=lr_orig_panel19,
                               thresh_arr=[thresh_arr[lr_orig_best_ind]])
        
        metrics, best_ind = describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[2, 'balanced_accuracy'], metrics['bal_acc'][best_ind], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'disparate_impact'], metrics['disp_imp'][best_ind], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'average_odds_difference'], metrics['avg_odds_diff'][best_ind], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'statistical_parity_difference'], metrics['stat_par_diff'][best_ind], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'equal_opportunity_difference'], metrics['eq_opp_diff'][best_ind], delta=self.tolerance)
        self.assertAlmostEqual(summary.at[2, 'theil_index'], metrics['theil_ind'][best_ind], delta=self.tolerance)
        ''' ------ '''
        
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


        
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
