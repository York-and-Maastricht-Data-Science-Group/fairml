# FairML: Towards Automated Fair Machine Learning

[Contents](#Contents)

* [Introduction](#Introduction)

* [Windows](#Windows)
  
  * [Windows-Installation](#Windows-Installation)
  * [Windows-Running](#Windows-Running)

* [Ubuntu](#Ubuntu)
  
  * [Ubuntu-Installation](#Windows-Installation)
  * [Ubuntu-Running](#Ubuntu-Running)

* [Docker](#Docker)
  
  * [Download from DockerHub](#Download-from-DockerHub)
  * [Build Dockerfile](#Build-Dockerfile)
  * [Running using Docker](#Running-using-Docker)

* [Tutorials](#Tutorials)

## Introduction

FairML is a tool that implements a model-based approach to model and automate bias measurement and mitigation in machine learning. 

1. FairML raises the abstraction of bias measurement and mitigation so that users can configure their bias mitigation model in YAML (YAML Ain't Markup Language), a human-friendly declarative language, without having to code in general/statistical programming languages.
2. It supports a certain degree of expressiveness, allowing users to experiment with different kinds of bias metrics, bias mitigation algorithms, datasets, classifiers, and their parameters to find the best combinations that reduce biases but with acceptable accuracy.
3. It automatically generates Python and Jupyter Notebook files which users can execute to run measure and mitigate biases on given datasets. All generated files are modifiable and extensible for fine-tuning and further development.

## Windows

### Windows-Installation

1. **Java JDK/JRE 11**. Make sure Java JDK/JRE 11 has been installed. FairML has only been tested  on OpenJDK 11.0.13 2021-10-19 and can be downloaded from https://adoptopenjdk.net/.

2. **Install Microsoft Visual Studio Build Desktop Development with C++ 2019**. Download Visual Studio Community 2022 setup/installer from https://visualstudio.microsoft.com/downloads/. Choose to install Microsoft Visual Studio Build Desktop Development with C++. This will also install Visual Studio Build Tools 2019.

3. **Install Anaconda**. Download and install Anaconda from https://www.anaconda.com/products/individual.

4. **Python dependencies**. Install all the following Python dependencies.
   
   ```
   pip install jupyter scipy numpy sklearn pandas tensorflow matplotlib aif360 shap fairlearn p2j adversarial-robustness-toolbox BlackBoxAuditing cvxpy numba 
   ```

5. **Install Maven**. Follow there instruction here https://maven.apache.org/install.html.

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

## Docker

### Download from DockerHub

The docker image can be found at [Fairml DockerHub](https://hub.docker.com/repository/docker/alfayohannisyorkacuk/fairml) or can be pulled using this command:

```
docker pull alfayohannisyorkacuk/fairml
```

### Build Dockerfile

You can also build the docker image manually. Download the **FairML project** from Github.

```
git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
```

Change directory to the source code of FairML generator:

```
cd fairml/generator/org.eclipse.epsilon.fairml.generator
```

And build the docker image using the command below. The building process takes 15 minutes on my computer (32 GB RAM, Intel i9).

```
docker build -t fairml .
```

### Running using Docker

#### Using Wizard

The command below will run FairML wizard (the '-w' argument). It will guide users to create a FairML model (*.flexmi) as well as its Python/Jupyter notebook implementation. The 'demo.flexmi' is the name of the FairML model file. This command generates demo.py and demo.ipynb files.

##### Windows

```
docker run --rm -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -w demo.flexmi
```

##### Ubuntu

```
docker run --rm -i -t -v $PWD:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -w demo.flexmi
```

#### Generate from a FairML model file

If you have an existing FairML model in flexmi file, execute the command below to generate its Python/Jupyter notebook implementation (the name of the file is the only argument). The 'demo.flexmi' is the name of the existing FairML model file. This command generates demo.py and demo.ipynb files. 

##### Windows

```
docker run --rm -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml demo.flexmi
```

##### Ubuntu

```
docker run --rm -i -t -v $PWD:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml demo.flexmi
```

#### Running the generated Jupyter Notebook file

Jupyter Notebook server is also embedded in  the Docker image. To run the generated Jupyter notebook file (**.ipynb*)  using the internal server, execute the command below. Use the '-j' argument and the name of ipynb file (demo.ipynb).

##### Windows

```
docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -j demo.ipynb
```

##### Ubuntu

```
docker run --rm -d -i -t -v $PWD:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -j demo.ipynb
```

Execute the command below to get the token for login later when accessing Jupyter notebook through your web browser.

```
docker exec fairml bash -c "jupyter notebook list"
```

It will show ouput similar to the following:

```
Currently running servers:
http://0.0.0.0:8888/?token=56ba87ebb9b359884b23f234bcf155d3afebf56f4323290c :: /fairml
```

Copy the token, and then use your web browser to access [http://localhost:8888](http://localhost:8888). Paste the token and the click the login button. Open the 'demo.ipyb' file and run it.

## Tutorials

Learn more about FairML in these tutorials.

1. [FairML Generator](docs/FairMLGenerator.md)
