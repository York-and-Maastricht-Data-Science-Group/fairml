# FairML Installation

### Windows-Installation

1. **Java JDK/JRE 11**. Make sure Java JDK/JRE 11 has been installed. FairML has only been tested on OpenJDK 11.0.13 2021-10-19 and can be downloaded from [https://adoptopenjdk.net/](https://adoptopenjdk.net/).

2. **Install Microsoft Visual Studio Build Desktop Development with C++ 2019**. Download Visual Studio Community 2022 setup/installer from [Download Visual Studio Tools - Install Free for Windows, Mac, Linux](https://visualstudio.microsoft.com/downloads/). Choose to install Microsoft Visual Studio Build Desktop Development with C++. This will also install Visual Studio Build Tools 2019.

3. **Install Anaconda**. Download and install Anaconda from https://www.anaconda.com/products/individual.

4. **Python dependencies**. Install all the following Python dependencies.
   
   ```
   pip install jupyter scipy numpy sklearn pandas tensorflow matplotlib aif360 shap fairlearn p2j adversarial-robustness-toolbox BlackBoxAuditing cvxpy numba 
   ```

5. **Install Maven**. Follow there instruction here [Maven &#x2013; Installing Apache Maven](https://maven.apache.org/install.html).

6. **Donwload the FairML project**. Download from Github.
   
   ```
   git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
   ```

7. **Build fairml.jar**. Follow the instructions below to build the file using Maven.
   
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

5. If you find errors while running it, there might be modules that haven't been installed yet. Install the modules using the 'pip' command.

## Ubuntu

### Ubuntu-Installation

1. **Java JDK/JRE 11**. Make sure Java JDK/JRE 11 has been installed. FairML has only been tested on OpenJDK 11.0.13 2021-10-19. It can be installed using this command.
   
   ```
   sudo apt-get install openjdk-11-jdk openjdk-11-jre
   ```

2. **Install Python 3, pip, and Maven**. Execute the commands below to install python3, pip, and maven.

```
sudo apt update
sudo apt install python3 python3-pip maven
```

3. **Python dependencies**. Install all the following dependencies. If it doesn't work, replace the 'pip3' with 'pip'. Ubuntu uses pip3 for Python 3 and pip for Python 2, but it depends on the settings of your local machine.

```
sudo pip3 install numba jupyter scipy numpy sklearn pandas tensorflow matplotlib aif360 shap fairlearn p2j adversarial-robustness-toolbox BlackBoxAuditing cvxpy numba 
```

4. **Download the FairML project**. Download from Github.

```
git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
```

5. **Build fairml.jar**. Follow the instructions below to build the file using Maven.

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

5. If you find errors while running it, there might be modules that haven't been installed yet. Install the modules using the 'pip3' or 'pip' command.
