# FairML Docker

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

The command below will run FairML wizard (the `-w` argument). It will guide users to create a FairML model (`*.flexmi`) as well as its Python/Jupyter notebook implementation. The 'demo.flexmi' is the name of the FairML model file. This command generates demo.py and demo.ipynb files.

##### Windows

```
docker run --rm -i -t -v %cd%:/fairml fairml -w demo.flexmi
```

##### Ubuntu

```
docker run --rm -i -t -v $PWD:/fairml fairml -w demo.flexmi
```

#### Generate from a FairML model file

If you have an existing FairML model in flexmi file, execute the command below to generate its Python/Jupyter notebook implementation (the name of the file is the only argument). The `demo.flexmi` is the name of the existing FairML model file. This command generates `demo.py` and `demo.ipynb` files.

##### Windows

```
docker run --rm -i -t -v %cd%:/fairml fairml demo.flexmi
```

##### Ubuntu

```
docker run --rm -i -t -v $PWD:/fairml fairml demo.flexmi
```

#### Running the generated Jupyter Notebook file

Jupyter Notebook server is also embedded in the Docker image. To run the generated Jupyter notebook file (`*.ipynb`) using the internal server, execute the command below. Use the `-j` argument and the name of ipynb file (`demo.ipynb`).

##### Windows

```
docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml-jupyter fairml -j demo.ipynb
```

##### Ubuntu

```
docker run --rm -d -i -t -v $PWD:/fairml --hostname=fairml -p 8888:8888 --name=fairml-jupyter fairml -j demo.ipynb
```

Use your web browser to access [http://localhost:8888](http://localhost:8888). Open the `demo.ipyb` file and run it.
