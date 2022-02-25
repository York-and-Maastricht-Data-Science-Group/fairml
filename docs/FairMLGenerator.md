# FairML Generator

[FairML Generator](#FairML-Generator)

* [Generating Jupyter Notebook files from Flexmi files](#Generating-Jupyter-Notebook-files-from-Flexmi-files)

* [Generating Jupyter Notebook files using Wizard](#Generating-Jupyter-Notebook-files-using-Wizard)

* [FairML Wizard](#FairML-Wizard)
  
  * [FairML Bias Mitigation Project](#FairML-Bias-Mitigation-Project)
  
  * [Dataset](#dataset)
  
  * [Bias Mitigation](#Bias-Mitigation)
  
  * [Classifier](#Classifier)
  
  * [Bias Mitigation Algorithms](#Bias-Mitigation-Algorithms)
  
  * [Bias Metrics](#Bias-Metrics)

FairML provides generator to help users produce 1.) Target Jupyter Notebook/Python file for bias mitigation and 2.) A wizard to help users generate Bias Mitigation model in [Epsilon Flexmi](https://www.eclipse.org/epsilon/doc/flexmi/#yaml-flavour) file in YAML format.  

## Generating Jupyter Notebook files from Flexmi files

In the *org.eclipse.epsilon.fairml.generator* directory, you will find 3 fairml files: fairml.jar, fairml.bat, and fairml.sh. Use *fairml.jar* if you want to generate Jupyter notebook file in Java way, while *fairml.bat* is for Windows and *fairml.sh* is for Ubuntu/Linux.  Both encapsulate the java command to execute the *fairml.jar* in each platform.

### Java

```
java -cp fairml.jar org.eclipse.epsilon.fairml.generator.FairML automated_selection.flexmi
```

The -cp argument in to define the classpath of  the package and Java class that are going to be excuted. In this case, the *fairml.jar* file and the full path of the main FairML class. The last argument is the name of the bias mitigation flexmi file, *automated_selection.flexmi*.

### Windows

```
fairml.bat automated_selection.flexmi
```

### Linux

```
fairml.sh automated_selection.flexmi
```

After executing the command, your console produces an output similar to this one.

```
Failed to load default content type configuration:
java.net.MalformedURLExceptionFailed to load default content type configuration:
java.net.MalformedURLExceptionFailed to load default content type configuration:
java.net.MalformedURLException
Notebook D:\PROJECTS\fairml\generator\org.eclipse.epsilon.fairml.generator\demo.ipynb written.
Finished!
```

Ignore the first four lines as there is a bug in the Epsilon Generation Language library, but it doesn't affect the quality of the output. Open the generated Jupyter notebook file  *automated_selection.ipynb* and run it.

## Generating Jupyter Notebook files using Wizard

FairML generator also provides wizard to guide users generate bias mitigation Jupyter notebook files.  To use wizard,  just add '-w' as another argument.

### Java

```
java -cp fairml.jar org.eclipse.epsilon.fairml.generator.FairML -w demo.flexmi
```

### Windows

```
fairml.bat automated_selection.flexmi -w demo.flexmi
```

### Linux

```
fairml.sh automated_selection.flexmi -w demo.flexmi
```

The demo.flexmi file is **the target** flexmi file, **not the source** of the bias mitigation model file. The wizard will prompt users with several questions to help users define the dataset, classifier for prediction, bias mitigation algorithms , and bias metrics in bias mitigation. 

## FairML Wizard

### FairML Bias Mitigation Project

```
=====================================
            FairML Wizard
=====================================
fairml 0.1

==== FairML ====
FairML project's name (default: Demo):
FairML project's description (default: Predict income <=50K: 0, >50K: 1):
```

Firstly, the wizard prompts the following lines to define the name and description of the FairML bias mitigation project. It also shows default values of the name and description if users do not provide them.

### Dataset

```
==== Dataset ====
Path to your dataset (default: data/adult.data.numeric.csv):
The name of the dataset: (default: Adult Dataset):
Predicted attribute (default: income-per-year):
Protected attributes (default: sex, race):
Categorical features (default: workclass, education, marital-status, occupation, relationship, native-country)
:
Train test split (default: 7, 2, 3):
```

The wizard also comes with the [adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult), a demo dataset, with labels have been transformed to numbers stored in the *data/adult.data.numeric.csv* file. The dataset is used as the default dataset in the wizard, and the following questions are based on the dataset. Users can input their own dataset, but all values should be in numbers/numeric, by the specifying the dataset's path, name, predicted attribute, protected attributes, and categorical features. Users also need to define the train, test, and validation split. The default ratio is 7 : 2 : 3 for training, testing, and validation.

### Bias Mitigation

```
==== Bias Mitigation ====
Bias mitigation's name (default Bias Mitigation 01):
Bias mitigation's description (default Bias Mitigation 01 Description):
```

After defining the dataset, users define the bias mitigation name and description. There could be more than one bias mitigation in a FairML project.

### Classifier

```
---- Training Algorithm ----
1. DecisionTreeClassifier
2. LogisticRegression
3. LinearSVC
Classifier (default 1. DecisionTreeClassifier):
```

After users choose the algorithm that they want to use in classification/prediction. By default, FairML uses *DecisionThreeClassifier*. Later, users can modify directly the FairML model in flexmi file or Jupyter nootebook file to define the parameters of the selected algorithms or add/use other classifiers other than the classifiers displayed in the wizard.

### Mitigation Algorithms

```
---- Mitigation Algorithm ----

# 1. Pre-processing
Apply bias mitigation in preprocessing (default: true):
The weights of the dataset are modifiable (default: true):
The bias mitigation allows latent space (default: false):

# 2. In-processing
Apply bias mitigation in in-processing (default: false):

# 3. Post-processing
Apply bias mitigation in post-processing (default: false):
```

Users are also askex to choose the debiasing algorithms for bias mitigation. By aswering the questions above, the generator automatically selects the approriate debiasing algorithms that are included in the generated Jupyter notebook file . Thus, reducing the time for users to select the appropriate debiasing algorithms, although they could manually added their own prefered debiasing algorithms later in the generated FairML model in flexmi file.

### Bias Metrics

```
---- Bias Metric ----
Measure group fairness (default: true):
Measure individual fairness (default: false):
Use single metrics for both individuals and groups (default: false):
Measure equal fairness (default: false):
Measure proportional fairness (default: false):
Measure false positives (default: false):
Measure false negatives (default: false):
Measure error rates (default: false):
Measure equal benefit (default: false):
```

Users then can choose what kinds of biases they want to measure. Later on, based on the users' responses, the generator automatically chooses the approriate bias metrics to be included in the generated Jupyter notebook files.

### Output

Below is an example of the FairML bias mitigation model generated by the wizard.  The configuration, particularly the 'true' and 'false' flags, is used to infer and automatically select the approriate debiasing algorithms and bias metrics to be included in the generated Jupyter notebook files.

```
?nsuri: fairml
fairml:
- name: Demo
- description: 'Predict income <=50K: 0, >50K: 1'
- dataset:
  - datasetPath: data/adult.data.numeric.csv
  - name: Adult Dataset
  - predictedAttribute: income-per-year
  - protectedAttributes: sex, race
  - categoricalFeatures: workclass, education, marital-status, occupation, relationship,
      native-country
  - trainTestSplit: 7, 2, 3
- biasMitigation:
  - name: Bias Mitigation 01
  - description: Bias Mitigation 01 Description
  - dataset: Adult Dataset
  - trainingMethod:
    - algorithm: DecisionTreeClassifier
    - parameters: criterion='gini', max_depth=4
  - prepreprocessingMitigation: 'true'
  - modifiableWeight: 'true'
  - allowLatentSpace: 'false'
  - inpreprocessingMitigation: 'false'
  - postpreprocessingMitigation: 'false'
  - groupFairness: 'true'
  - individualFairness: 'false'
  - groupIndividualSingleMetric: 'false'
  - equalFairness: 'false'
  - proportionalFairness: 'false'
  - checkFalsePositive: 'false'
  - checkFalseNegative: 'false'
  - checkErrorRate: 'false'
  - checkEqualBenefit: 'false'
```
