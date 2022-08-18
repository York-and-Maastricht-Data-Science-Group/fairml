# FairML: Towards Automated Fair Machine Learning

## [Archival Repository](#Contents)
FairML is permanently available at Zenodo: https://doi.org/10.5281/zenodo.7007839.

## [Contents](#Contents)

- [FairML: Towards Automated Fair Machine Learning](#fairml-towards-automated-fair-machine-learning)
  - [Archival Repository](#archival-repository)
  - [Contents](#contents)
  - [Authors](#authors)
  - [Abstract](#abstract)
  - [Archive Contents](#archive-contents)
  - [Step-by-step Setup/Installation Guide](#step-by-step-setupinstallation-guide)
    - [More...](#more)
  - [Others](#others)
    - [Videos](#videos)
    - [Tutorials](#tutorials)
    - [Installation](#installation)
    - [Docker](#docker)
  - [Reproducing the Evaluation](#reproducing-the-evaluation)
    - [Generating Python and Jupyter Notebook Files](#generating-python-and-jupyter-notebook-files)
    - [Measuring the Generation Time](#measuring-the-generation-time)
    - [Generated vs Original Execution Time](#generated-vs-original-execution-time)
    - [Measuring the Correctness](#measuring-the-correctness)
    - [Original Lines of Code (LoC) vs Model LoC  vs Generated LoC](#original-lines-of-code-loc-vs-model-loc--vs-generated-loc)

## [Authors](#contents)

- Alfa Yohannis (alfa.yohannis@york.ac.uk)
- Dimitris Kolovos (dimitris.kolovos@york.ac.uk)

## [Abstract](#contents)
Models produced by machine learning are not guaranteed to be free from bias, particularly when trained and tested with data produced in discriminatory environments. The bias can be unethical, mainly when the data contains sensitive attributes, such as sex, race, age, etc. Some approaches have contributed to mitigating such biases by providing bias metrics and mitigation algorithms. The challenge is that users have to implement their code in general/statistical programming languages, which can be demanding for users with little experience in programming and fairness in machine learning.

FairML is a tool that implements a model-based approach to facilitate bias measurement and mitigation in machine learning with the reduced software development effort.

1. FairML raises the abstraction of bias measurement and mitigation so that users can configure their bias mitigation model in YAML (YAML Ain't Markup Language), a human-friendly declarative language, without having to code in general/statistical programming languages.
2. It supports a certain degree of expressiveness, allowing users to experiment with different bias metrics, bias mitigation algorithms, datasets, classifiers, and their parameters to find the best combinations that reduce biases but with acceptable accuracy.
3. It automatically generates Python and Jupyter Notebook files which users can execute to measure and mitigate biases on given datasets. All generated files are modifiable and extensible for fine-tuning and further development.

## [Archive Contents](#contents)

The archive contains:

1. A preprint version of the accepted paper.
2. This README.md file.
3. Other related *.md files in the `docs` subdirectory.
4. The source code of FairML, but it's better to directly download it from https://github.com/York-and-Maastricht-Data-Science-Group/fairml to use the most recent version. 

## [Step-by-step Setup/Installation Guide](#contents)

The fastest way to run and test FairML is by using its docker image. The video can be found at https://tinyurl.com/mrydfed9.

1. Open your command prompt and execute the following command. First, we create a directory where we want to put your work. Let's say that we want to work under the FAIRML directory.
   
   ```
   mkdir FAIRML
   cd FAIRML
   ```

2. We pull the docker image from DockerHub. 
   
   ```
   docker pull alfayohannisyorkacuk/fairml
   ```

3. Then, we use the FairML Wizard to create a FairML model saved in a flexmi file by executing the command below on **Windows**. **For Linux, replace the %cd% with $PWD**. They are the environment variables for the current directory. The `-w` flag indicates to run a wizard.
   
   ```
   docker run --rm -i -t -v %cd%:/fairml alfayohannisyorkacuk/fairml -w demo.flexmi
   ```

4. **The Wizard prompts us with some questions**. Just **choose all the default values, for now,**, EXCEPT for the `Measure equal fairness (default: false):` question. Please type `true` and enter. For the rest, keep selecting their default values.
   
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
   
   Your answers to the questions help the Wizard to select the best bias mitigation algorithms and metrics.

5. After the end of the Wizard, the following directories and files are generated. Depending on your operating system, use `dir` or `ls` commands to see the generated files and directories.
   
   ```
   data
   demo.flexmi
   demo.ipynb
   demo.py
   fairml.py
   generator
   ```

6. Let's run Jupyter Notebook to execute the generated `demo.ipynb`. Remember to replace `%cd` with `$PWD` if running on Linux.
   
   ```
   docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml-jupyter alfayohannisyorkacuk/fairml -j demo.ipynb
   ```

7. Use your browser to access [http://localhost:8888](http://localhost:8888). You will see Jupyter Notebook with your current directory as your working directory. Open the `demo.ipynb` file and run the whole notebook.

### [More...](#contents)

8. We don't have to run the Wizard all the time. We can directly add the desired classifiers, bias mitigation algorithms, and bias metrics directly into the generated `*.flexmi` file and regenerate the `*.py`/`*.ipynb` files. Let's add the `theil_index` metric to the end of the `demo.flexmi` file, so the end part becomes like this.
   
   ```
     ...
     ...
     - prepreprocessingMitigation: 'true'
     - equalFairness: 'true'
     - biasMetric: 
       - name: theil_index
   ```

9. Run the following command, without the `-w` flag, to regenerate the `*.py`/`*.ipynb` files.
   
   ```
   docker run --rm -i -t -v %cd%:/fairml alfayohannisyorkacuk/fairml demo.flexmi
   ```

10. Refresh your Jupyter Notebook tab associated with the `demo.ipynb` file on your browser and rerun your whole notebook. You will notice that the `theil_index` metric has been added to your Jupyter notebook.

11. Feel free to check and modify the generated `*.flexmi`, `*.py`, `*.ipynb` files to learn more about FairML and modify the results.

12. Don't forget to terminate the running Docker container using the following command.
    ```
    docker stop fairml-jupyter
    ```

## [Others](#contents)

Some other stuff to explore.

### [Videos](#contents)
FairML's video tutorials can be found here https://tinyurl.com/mrydfed9 or https://drive.google.com/drive/folders/1V1NrEN1fghbwzUmR-urEzhcZPvTdqJLC?usp=sharing.

### [Tutorials](#contents)

Learn more about FairML using these tutorials.

1. [FairML Generator](docs/Generator.md)
2. [FairML Model](docs/Model.md)

### [Installation](#contents)

For installation &ndash; setting up an environment, downloading and building source code, please follow the instructions here [FairML Installation](docs/Installation.md).

### [Docker](#contents)

For more detailed documentation for downloading, building, and running FairML docker images, please follow the instructions here [FairML Docker](docs/Docker.md).

## [Reproducing the Evaluation](#contents)

This section contains the instructions to reproduce the evaluation results presented in the paper. You can also watch the videos here https://tinyurl.com/mrydfed9.

### [Generating Python and Jupyter Notebook Files](#contents)

1. Make sure to perform the [instalation instruction](docs/Installation.md) first before proceeding to the next steps.

2. Download FairML source code using the command below.
   
   ```
   git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
   ```

3. Use Eclipse, open the `org.eclipse.epsilon.fairml.generator` project located inside the `generator` directory. It is a **Maven** project. So, all dependencies will be automatically downloaded.

4. Inside the project, there is the `test-model` directory. It contains all the FairML models used in the evaluation. We can use any of the following commands to generate the target Python/Jupyter notebook files one-by-one per model. 
   
   ```
   # On Linux
   ./fairml.sh test-model/paper_demo.flexmi 
   # On Windows
   ./fairml.sh test-model\paper_demo.flexmi
   # Or using Java
   java -cp "fairml.jar" org.eclipse.epsilon.fairml.generator.FairML test-model/paper_demo.flexmi
   ```
   
    However, it's more efficient to run the JUnit test to generate all the target models. Under the `src` directory, inside the `org.eclipse.epsilon.fairml.generator.test` package, open the `AIF360ExampleTest.java` unit test file. Run the file as a JUnit test. All the target files will be generated inside the `test-model` directory.

5. Get into the `test-model` directory. Run the following command to open the target output in Jupyter Notebook:
   
   ```Bat
   docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml-jupyter alfayohannisyorkacuk/fairml -j paper_demo.flexmi
   ```

6. Copy the `load_preproc_data_adult.csv` in the `data` directory to `test-model/data` directory. 

7. Browse http://localhost:8080. Open the `paper_demo.ipynb` and run all (the whole notebook).

8. Don't forget to terminate the running Docker container using the command below.
   
   ```
   docker stop  fairml-jupyter
   ```

9. Feel free to modify the FairML model files and regenerate the target files or create a new FairML model file from scratch or generate using the [Wizard](docs/Generator.md).

### [Measuring the Generation Time](#contents)

1. Under the `src` directory, inside the `org.eclipse.epsilon.fairml.generator.test` package, open the `AIF360ExampleTest.java` unit test file.

2. Comment the following lines:
   
   ```Python
   int startMeasure = 1;
   int endMeasure = 1;
   ```
   
    And uncomment the following lines:
   
   ```Python
   //   int startMeasure = 5;
   //   int endMeasure = 14;
   ```

3. Run the file as a JUnit test. All the target files will be generated inside the `test-model` directory. It would take some time to run all the tests in 14 iterations. The end results are the measured generation time for each test model.

### [Generated vs Original Execution Time](#contents)

This evaluation requires all the target Python/Jupyter notebook files to be generated first (check [Generating Python and Jupyter Notebook Files](#Generating-Python-and-Jupyter-Notebook-Files)). 

Also, you need to download the datasets required by the IBM Fairness AI 360 library manually. Go to the installation directory of the library. In my local machines, it's located in `C:\Anaconda3\Lib\site-packages\aif360\data\raw\meps`. Therefore change the active directory in the directory:
```
cd C:\Anaconda3\Lib\site-packages\aif360\data\raw\meps
```
For more information, read the `README.md` inside the directory. You also need to install R, the statistical programming language. Then execute the following command to download the datasets:
```
"C:\Program Files\R\R-4.2.1\bin\x64\Rscript.exe" generate_data.R
```
You also need to download other datasets for every directory under:
```
C:\Anaconda3\Lib\site-packages\aif360\data\raw
```
Please visit every directory inside it and read every `README.md` to download the raw data for each directory.

There is also bug in `C:\Anaconda3\Lib\site-packages\aif360\explainers\metric_json_explainer.py` which the string at line 147 should one line with the string at line 146. Please fix it manually.



This evaluation compares the execution time of the generated target files vs the original files in IBM Fairness AI 360(under the `ibmfai360` directory or at https://github.com/Trusted-AI/AIF360/tree/master/examples). The perform the evaluation, run the `execution_time_performance.py` Python file under the `test-python` directory. It will take some time to finish all the examples. Every example runs 14 iterations. If you just want to get through all the examples once, comment all values under the `multiple iteration` line, and use the values under the `single iteration` line.

```Python
## multiple iteration
# threshold = 5 #5
# fr = 1 #1
# to = 14 #14

## single iteration
threshold = 0 #5
fr = 1 #1
to = 2 #14
```

### [Measuring the Correctness](#contents)

This evaluation also requires all the target Python/Jupyter notebook files to be generated first (check [Generating Python and Jupyter Notebook Files](#generating-python-and-jupyter-notebook-files)). To evaluate the correctness of the target Python/Jupyter notebook files, run the `fairml_test.py` unit test file in the `test-python` directory. The file contains the representations of the original target files in the IBM AI Fairness 360 examples. It performs `assertAlmostEqual` assertions with 0.1 tolerance at many measurement points to compare that generated target files produce the same values as in their respective original files.

### [Original Lines of Code (LoC) vs Model LoC  vs Generated LoC](#contents)

This evaluation also requires all the target Python/Jupyter notebook files to be generated first (check [Generating Python and Jupyter Notebook Files](#generating-python-and-jupyter-notebook-files)). 

We use [cloc](http://cloc.sourceforge.net/) for this evaluation. 
For the **original code** and the **generated code**, we calculate the LoCs of the files in the `ibmfai360` and `test-model` directories and respectively, using the following command.

```
cloc . --by-file --force-lang=python --include-ext=py
```

For the **model code**, we execute the following command.

```
cloc . --by-file --force-lang=yaml --include-ext=flexmi
```

