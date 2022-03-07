# FairML Model

## Contents
- [FairML Model](#fairml-model)
  - [Contents](#contents)
  - [FairML](#fairml)
    - [Dataset](#dataset)
    - [Bias Mitigation](#bias-mitigation)
      - [Training Methods](#training-methods)
      - [Mitigation Methods](#mitigation-methods)
      - [Bias Metrics](#bias-metrics)

## [FairML](#contents)
Below is an example of a FairML bias mitigation model. The file of the model has `.flexmi` extension and formatted in YAML flavour. The `fairml` tag is the **root** component. It has `name` and `description` attributes to describe the FairML bias mitigation model. The name of the model below is `Explainer` and the description contains a link pointing to an external Jupyter notebook file.

```yaml
?nsuri: fairml
fairml:
- name: Explainer
- description: |-
    Test Explainer using German Credit Dataset
    https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_credit_scoring.ipynb

# set the dataset
- dataset:
  - name: German Credit Dataset
  - datasetPath: ../data/german.numeric.csv
  - predictedAttribute: credit
  - protectedAttributes: mature 
  - trainTestValidationSplit: 7, 3
  - droppedAttributes: personal_status, sex, age
  - categoricalFeatures: |-
     status, credit_history, purpose,
     savings, employment, other_debtors, property,
     installment_plans, housing, skill_level, telephone,
     foreign_worker

# define the bias mitigation
- biasMitigation:
  - name: Test Explainer using German Credit Dataset  
  - dataset: German Credit Dataset
  - equalFairness: true
  
  - trainingMethod:
    - algorithm: DecisionTreeClassifier
    - parameters: criterion='gini', max_depth=4
  - trainingMethod:
    - algorithm: LinearSVC
      
  - mitigationMethod:
    - algorithm: Reweighing
  - mitigationMethod:
    - algorithm: MetaFairClassifier
  
  - biasMetric:
    - name: accuracy
  - biasMetric: 
    - name: theil_index
```
### [Dataset](#contents)

In the model, we could define the datasets for bias mitigation. We could define its name, file path, predicted attribute, protected attributes, train-test-validation split, attributes to be dropped, and categorical features. We could define many datasets and use them in bias mitigation.

```yaml
# set the dataset
- dataset:
  - name: German Credit Dataset
  - datasetPath: ../data/german.numeric.csv
  - predictedAttribute: credit
  - protectedAttributes: mature 
  - trainTestValidationSplit: 7, 3
  - droppedAttributes: personal_status, sex, age
  - categoricalFeatures: |-
     status, credit_history, purpose,
     savings, employment, other_debtors, property,
     installment_plans, housing, skill_level, telephone,
     foreign_worker
```

### [Bias Mitigation](#contents)

A FairML model could have more than one bias mitigation, but here we only show one. In a bias mitigation, we could define its name, dataset that we want to use (in this case, the German Credit Dataset defined above), and several flags that could help FairML generator to automatically select the best bias mitigation algorithms and bias metrics for our cases. In the model, we set equal fairness to `true`. This setting would make FairML generator automatically adds `disparate_impact` and `statistical_parity_difference` as  bias metrics when generating target `*.py`/`*.ipynb` files. 

```yaml
- biasMitigation:
  - name: Test Explainer using German Credit Dataset  
  - dataset: German Credit Dataset
  - equalFairness: true
```

#### [Training Methods](#contents)

Every bias mitigation could have many training methods or classifiers. We can define the algorithm of each training method as well as the parameters for the algorithm. In the model, we use two algorithms. The first one is  `DecisionTreeClassifier` with parameters `criterion='gini'` and `max_depth=4`. The second algorithm is `LinearSVC`. 

```yaml
  - trainingMethod:
    - algorithm: DecisionTreeClassifier
    - parameters: criterion='gini', max_depth=4
  - trainingMethod:
    - algorithm: LinearSVC
```

#### [Mitigation Methods](#contents)

In bias mitigation, we could also define the bias mitigation methods that we want to use to reduce biases. In the model, we use two algorithms for debiasing, `Reweighing` (pre-processing debiasing) and `MetaFairClassifier` (in-processing debiasing).
```yaml
  - mitigationMethod:
    - algorithm: Reweighing
  - mitigationMethod:
    - algorithm: MetaFairClassifier
```

#### [Bias Metrics](#contents)
The last part of bias mitigation is bias metrics. In the model, we explicitly define two bias metrics, `accuracy` and `theil_index`. 

```yaml  
  - biasMetric:
    - name: accuracy
  - biasMetric: 
    - name: theil_index
```