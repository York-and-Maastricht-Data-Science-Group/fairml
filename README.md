# FairML: Towards Automated Fair Machine Learning


## [Contents](#Contents)

- [FairML: Towards Automated Fair Machine Learning](#fairml-towards-automated-fair-machine-learning)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Up and Running](#up-and-running)
  - [Others](#others)
    - [Tutorials](#tutorials)
    - [Installation](#installation)
    - [Docker](#docker)

## [Introduction](#contents)

FairML is a tool that implements a model-based approach to model and automate bias measurement and mitigation in machine learning. 

1. FairML raises the abstraction of bias measurement and mitigation so that users can configure their bias mitigation model in YAML (YAML Ain't Markup Language), a human-friendly declarative language, without having to code in general/statistical programming languages.
2. It supports a certain degree of expressiveness, allowing users to experiment with different kinds of bias metrics, bias mitigation algorithms, datasets, classifiers, and their parameters to find the best combinations that reduce biases but with acceptable accuracy.
3. It automatically generates Python and Jupyter Notebook files which users can execute to run measure and mitigate biases on given datasets. All generated files are modifiable and extensible for fine-tuning and further development.


## [Up and Running](#contents)

The fastest way to run and test FairML is by using its docker image.

1. Open your command prompt and execute the following command. First, we create a directory where we want to put your work. Let's say that we want work under FAIRML directory.
   ```
   mkdir FAIRML
   cd FAIRML
   ```
2. We pull the docker image from DockerHub. 
   ```
   docker pull alfayohannisyorkacuk/fairml
   ```
3. Then, we use the FairML wizard to create an FairML model saved in a flexmi file by executing the command below on **Windows**. **For Linux, replace the %cd% with $PWD**. They are the environment variables for current directory. The `-w` flag indicates to run a wizard.
   ```
   docker run --rm -i -t -v %cd%:/fairml fairml -w demo.flexmi
   ```
4. **The wizard prompt us with some questions**. Just **choose all the default values for now**, EXCEPT for the `Measure equal fairness (default: false):` question. Please type `true` and enter. For the rest, keep select their default values.
   ```:
   =====================================
            FairML Wizard
    =====================================
    fairml 0.1
    ==== FairML ====
    FairML project's name (default: Demo):
    ...
    ...
    Measure equal fairness (default: false): true
    ...
    ...
   ```
   
   Your answers to the questions help the wizard to select the best bias mitigation algorithms and metrics.

5. After the end of the wizard, the following directories and files are generated. Use `dir` or `ls` commands, depedending on your operating systems, to see the generated files and directories.
   ```
   data
   demo.flexmi
   demo.ipynb
   demo.py
   fairml.py
   generator
   ```
6. Let's run Jupyter Notebook to execute the generated `demo.ipynb`. Remember to replace `%cd` with `$PWD` if you are running on Linux.
   ```
   docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml-jupyter fairml -j demo.ipynb
    ```
7. Use your browser to access [http://localhost:8888](http://localhost:8888). You will see Jupyter Notebook with your current directory as your working directory. Open `demo.ipynb` file and run the whole notebook.

### [More...](#contents)

8. We don't have to run the wizard all the time. We can directly add the desired classifiers, bias mitigation algorithms, and bias metrics directly into the generated `*.flexmi` file and re-generate the `*.py`/`*.ipynb` files. Let's add `theil_index` metric to the end of `demo.flexmi` file so the end of file becomes like this.
   ```
     ...
     ...
     - prepreprocessingMitigation: 'true'
     - equalFairness: 'true'
     - biasMetric: 
       - name: theil_index
   ```
9. Run the following command, without the `-w` flag, to re-generate the `*.py`/`*.ipynb` files.
   ```
   docker run --rm -i -t -v %cd%:/fairml fairml demo.flexmi
   ```
10. Refresh your Jupyter Notebook tab on associated with `demo.ipynb` file on your brower and run your whole notebook again. You will notice that `theil_index` metric has been added to your notebook.
11. Feel free to check and modify the generated `*.flexmi`, `*.py`, `*.ipynb` files to learn more about FairML and modify the results.

## [Others](#contents)

### Tutorials

Learn more about FairML using these tutorials.

1. [FairML Generator](docs/FairMLGenerator.md)

### Installation

For installation &ndash; setting up environment, downloading and building source code, please follow the instructions here [FairML Installation](docs/Installation.md).

### Docker

For a more detailed documentation for downloading, building, and running FairML docker image, please follow the instructions here [FairML Docker](docs/Docker.md).


