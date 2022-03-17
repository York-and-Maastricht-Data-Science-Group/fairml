'''
Created on 1 Mar 2022

@author: Alfa Yohannis
'''
import sys
sys.path.insert(0, '../test-model')

import unittest


class Test(unittest.TestCase):

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

        self.assertAlmostEqual(summary.at[1, 'mean_difference'],
                               metric_orig_train.mean_difference(), places=4)
        self.assertAlmostEqual(summary.at[2, 'mean_difference'],
                               metric_transf_train.mean_difference(), places=4)
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
        self.assertAlmostEqual(summary.at[2, 'rich_subgroup'],
                               gamma_disparity, places=4)
        
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
        
            pos = 3
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
                               fp_violations[0], places=4)
                pos = pos + 1
                
            if save_results:
                pickle.dump(results_dict, open('results_dict_' + str(gamma_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))
        
        multiple_classifiers_pareto(data_set)
        print("")


    def test_demo_optimized_preprocessing_adult(self):
        
        # import from the generated python code 
        import demo_optimized_preprocessing_adult
        summary = demo_optimized_preprocessing_adult.fairml.bias_mitigations[0].summary_table
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
        
        privileged_groups = [{'race': 1}] # White
        unprivileged_groups = [{'race': 0}] # Not white
        
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary.at[1, 'mean_difference'],
                               metric_orig_train.mean_difference(), places=4)
        
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
                               metric_transf_train.mean_difference(), places=4)
        
    
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
        self.assertAlmostEqual(summary1.at[1, 'mean_difference'], metric_orig_test.mean_difference(), places=4)
        ''' ------ '''
        
        biased_model = MetaFairClassifier(tau=0, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)
        
        
        dataset_bias_test = biased_model.predict(dataset_orig_test)
        
        
        classified_metric_bias_test = ClassificationMetric(dataset_orig_test, dataset_bias_test,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = {:.6f}".format(classified_metric_bias_test.accuracy()))
        TPR = classified_metric_bias_test.true_positive_rate()
        TNR = classified_metric_bias_test.true_negative_rate()
        bal_acc_bias_test = 0.5*(TPR+TNR)
        print("Test set: Balanced classification accuracy = {:.6f}".format(bal_acc_bias_test))
        print("Test set: Disparate impact = {:.6f}".format(classified_metric_bias_test.disparate_impact()))
        fdr = classified_metric_bias_test.false_discovery_rate_ratio()
        #fdr = min(fdr, 1/fdr)
        print("Test set: False discovery rate ratio = {:.6f}".format(fdr))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[2, 'accuracy'], classified_metric_bias_test.accuracy(), places=4)
        self.assertAlmostEqual(summary1.at[2, 'balanced_accuracy'], bal_acc_bias_test, places=4)
        self.assertAlmostEqual(summary1.at[2, 'disparate_impact'], classified_metric_bias_test.disparate_impact(), places=4)
        self.assertAlmostEqual(summary1.at[2, 'false_discovery_rate_ratio'], classified_metric_bias_test.false_discovery_rate_ratio(), places=4)
        ''' ------ '''
        
        debiased_model = MetaFairClassifier(tau=0.7, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)
        
        
        dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
        
        
        metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {:.6f}".format(metric_dataset_debiasing_test.mean_difference()))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[3, 'mean_difference'], metric_dataset_debiasing_test.mean_difference(), places=4)
        ''' ------ '''
        
        classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test, 
                                                         dataset_debiasing_test,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = {:.6f}".format(classified_metric_debiasing_test.accuracy()))
        TPR = classified_metric_debiasing_test.true_positive_rate()
        TNR = classified_metric_debiasing_test.true_negative_rate()
        bal_acc_debiasing_test = 0.5*(TPR+TNR)
        print("Test set: Balanced classification accuracy = {:.6f}".format(bal_acc_debiasing_test))
        print("Test set: Disparate impact = {:.6f}".format(classified_metric_debiasing_test.disparate_impact()))
        fdr = classified_metric_debiasing_test.false_discovery_rate_ratio()
        #fdr = min(fdr, 1/fdr)
        print("Test set: False discovery rate ratio = {:.6f}".format(fdr))
        
        ''' ASSERT '''
        self.assertAlmostEqual(summary1.at[3, 'accuracy'], classified_metric_debiasing_test.accuracy(), places=4)
        self.assertAlmostEqual(summary1.at[3, 'balanced_accuracy'], bal_acc_debiasing_test, places=4)
        self.assertAlmostEqual(summary1.at[3, 'disparate_impact'], classified_metric_debiasing_test.disparate_impact(), places=4)
        self.assertAlmostEqual(summary1.at[3, 'false_discovery_rate_ratio'], classified_metric_debiasing_test.false_discovery_rate_ratio(), places=4)
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
        self.assertAlmostEqual(summary2.at[2, 'accuracy'], accuracies[0], places=4)
        self.assertAlmostEqual(summary2.at[2, 'disparate_impact'], disparates[0], places=4)
        
        gen_accuracies = summary2["accuracy"].values.tolist()[1:11]
        gen_impacts = summary2["disparate_impact"].values.tolist()[1:11]
        avg_gen_accuracies = sum(gen_accuracies) / len(gen_accuracies) 
        avg_gen_impacts = sum(gen_impacts) / len(gen_impacts) 
        avg_demo_accuracies = sum(accuracies) / len(accuracies) 
        avg_demo_impacts = sum(disparates) / len(disparates) 
        
        self.assertAlmostEqual(avg_gen_accuracies, avg_demo_accuracies, delta=0.1)
        self.assertAlmostEqual(avg_gen_impacts, avg_demo_impacts, delta=0.1)
        
        ''' ------ '''
        
        fig, ax1 = plt.subplots(figsize=(13,7))
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
            self.assertAlmostEqual(summary.at[x + 2, 'disparate_impact'], val, places=4)
            ''' ------ '''

            x = x + 1
        
        plt.plot(np.linspace(0, 1, 11), DIs, marker='o')
        plt.plot([0, 1], [1, 1], 'g')
        plt.plot([0, 1], [0.8, 0.8], 'r')
        plt.ylim([0.0, 1.2])
        plt.ylabel('Disparate Impact (DI)')
        plt.xlabel('repair level')
        plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
