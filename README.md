# FairML: Towards Automated Fair Machine Learning
[Contents](#Contents)
* [Introduction](#Introduction)
* [Windows](#Windows)
  * [Windows-Installation](#Windows-Installation)
  * [Windows-Running](#Windows-Running)
* [Ubuntu](#Ubuntu)
  * [Ubuntu-Installation](#Windows-Installation)
  * [Ubuntu-Running](#Ubuntu-Running)
## Introduction

FairML is a tool that implements a model-based approach to model and automate bias measurement and mitigation in machine learning. 

1. FairML raises the abstraction of bias measurement and mitigation so that users can configure their bias mitigation model in YAML (YAML Ain't Markup Language), a human-friendly declarative language, without having to code in general/statistical programming languages.
2. It supports a certain degree of expressiveness, allowing users to experiment with different kinds of bias metrics, bias mitigation algorithms, datasets, classifiers, and their parameters to find the best combinations that reduce biases but with acceptable accuracy.
3. It automatically generates Python and Jupyter Notebook files which users can execute to run measure and mitigate biases on given datasets. All generated files are modifiable and extensible for fine-tuning and further development.

## Windows

### Windows-Installation

1. **Install Microsoft Visual Studio Build Desktop Development with C++ 2019**. Download Visual Studio Community 2022 setup/installer from https://visualstudio.microsoft.com/downloads/. Choose to install Microsoft Visual Studio Build Desktop Development with C++. This will also install Visual Studio Build Tools 2019.

2. **Install Anaconda**. Download and install Anaconda from https://www.anaconda.com/products/individual.

3. **Python dependencies**. Install all the following Python dependencies.

   ```
   pip install jupyter scipy numpy sklearn pandas tensorflow matplotlib aif360 shap fairlearn p2j
   ```

4. **Install Maven**. Follow there instruction here https://maven.apache.org/install.html.

5. **Donwload the FairML project**. Download from Github.

   ```
   git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
   ```

6. **Build fairml.jar**. Follow the instructions below to build the file using Maven.

   ```
   cd fairml\generator\org.eclipse.epsilon.fairml.generator
   mvn install
   ```

   You will find **fairml.jar** file created.

### Windows-Running

1. Inside the **fairml\generator\org.eclipse.epsilon.fairml.generator** directory, you will can find **automated_selection.flexmi** file. The file represents FairML model in *.flexmi extension expressed in YAML language. You can also find **fairml.bat**. The file is a helper file to generate Bias Mitigation code in Python and Jupyter Notebook files in **Windows**.

2. execute the following command to generate the files.

   ```
   fairml.bat automated_selection.flexmi
   ```

   The command will produce four files:

   ```
   automated_selection.flexmi.xmi
   automated_selection.ipynb
   automated_selection.py
   fairml.py
   ```

3. Run Jupyter Notebook.

   ```
   >> jupyter notebook
   ```

4. Open the **automated_selection.ipynb** in Jupyter Notebook and run the whole notebook.

4. If you find errors while running it, there might be modules that haven't been installed yet. Install the modules using the 'pip' command.

## Ubuntu

### Ubuntu-Installation

1. **Install Python 3, pip, and Maven**. Execute the commands below to install python3, pip, and maven.

   ```
   sudo apt update
   sudo apt install python3 python3-pip maven
   ```

2. **Python dependencies**. Install all the following dependencies. If it doesn't work, replace the 'pip3' with 'pip'. Ubuntu uses pip3 for Python 3 and pip for Python 2, but it depends on the settings of your local machine.

   ```
   pip3 install jupyter scipy numpy sklearn pandas tensorflow matplotlib aif360 shap fairlearn p2j
   ```

3. **Download the FairML project**. Download from Github.

   ```
   git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
   ```

4. **Build fairml.jar**. Follow the instructions below to build the file using Maven.

   ```
   cd fairml/generator/org.eclipse.epsilon.fairml.generator
   mvn install
   ```

   You will find **fairml.jar** file created.

### Ubuntu-Running

1. Inside the **fairml\generator\org.eclipse.epsilon.fairml.generator** directory, you can find **automated_selection.flexmi** file. The file represents FairML model in *.flexmi extension expressed in YAML language. You can also find **fairml.sh**. The file is a helper file to generate Bias Mitigation code in Python and Jupyter Notebook files in **Ubuntu**.

2. Set the fairml.sh file to be executable, and execute the command to generate the files.

   ```
   sudo chmod +x fairml.sh
   ./fairml.sh automated_selection.flexmi
   ```

   The command will produce four files:

   ```
   automated_selection.flexmi.xmi
   automated_selection.ipynb
   automated_selection.py
   fairml.py
   ```

3. Run Jupyter Notebook.

   ```
   >> jupyter notebook
   ```

4. Open the **automated_selection.ipynb** in Jupyter Notebook and run the whole notebook.

4. If you find errors while running it, there might be modules that haven't been installed yet. Install the modules using the 'pip3' or 'pip' command.
