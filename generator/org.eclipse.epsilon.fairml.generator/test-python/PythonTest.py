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


    def testTutorialCreditScoring(self):
        import numpy as np
        np.random.seed(0)
        
        from aif360.datasets import GermanDataset
        from aif360.metrics import BinaryLabelDatasetMetric
        from aif360.algorithms.preprocessing import Reweighing
        
        from IPython.display import Markdown, display
        
        dataset_orig = GermanDataset(
            protected_attribute_names=['age'],           # this dataset also contains protected
                                                         # attribute for "sex" which we do not
                                                         # consider in this evaluation
            privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
            features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
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
        fairml = tutorial_credit_scoring.fairml
        av1 = round(fairml.bias_mitigations[0].summary_table.at[1, 'mean_difference'], 3)
        av2 = round(fairml.bias_mitigations[0].summary_table.at[2, 'mean_difference'], 3)
        
        delta1 = abs(av1 - round(metric_orig_train.mean_difference(), 3))
        delta2 = abs(av2 - round(metric_transf_train.mean_difference(), 3))
    
        self.assertLessEqual(delta1, 0.01)
        self.assertLessEqual(delta2, 0.01)

        print("")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()