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
        
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
