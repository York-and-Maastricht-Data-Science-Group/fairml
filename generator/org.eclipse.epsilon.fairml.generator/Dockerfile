# docker stop fairml-jupyter
# docker rm fairml
# docker rmi fairml
# docker system prune

# docker image build . -t fairml
# docker tag fairml:latest alfayohannisyorkacuk/fairml:latest
# docker push alfayohannisyorkacuk/fairml:latest


# run the following command to use this docker container later after installing
# docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -w demo.flexmi
# docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml demo.flexmi
# docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -j demo.ipynb
# docker exec fairml bash -c "jupyter notebook list"
# docker exec -i fairml bash -c "python3 -m notebook.auth password"
# jupyter notebook demo.ipynb --port=8888 --no-browser --ip='*' --allow-root --NotebookApp.password_required=False --NotebookApp.allow_remote_access=True --NotebookApp.token='' --NotebookApp.password=''
# FROM openjdk:11-jdk-slim-bullseye

# STAGE 01: Do all the Java stuff to build only the fairml
FROM openjdk:11-jdk-slim-bullseye AS stage01
# FROM debian:bullseye-slim AS Stage01

WORKDIR /fairml-src

RUN apt-get update -y && \
  # apt-get install -y openjdk-11-jre-headless nocache && \
  apt-get install -y git nocache && \
  apt-get install -y maven nocache && \
  git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git && \
  cd /fairml-src/fairml/generator/org.eclipse.epsilon.fairml.generator && \
  apt-get remove -y git nocache && \
  apt-get autoremove -y

WORKDIR /fairml-src/fairml/generator/org.eclipse.epsilon.fairml.generator

RUN mvn install && \
  mkdir /fairml && \
  cp -v fairml.jar /fairml-src && \
  apt-get remove -y maven nocache && \
  apt-get autoremove -y && \
  rm -rf /fairml-src/fairml && \
  rm -rf /root/.m2/repository 

# STAGE 02: copy the fairml in stage 2 and install all python dependencies
# FROM debian:bullseye-slim
FROM openjdk:11-jre-slim-bullseye

WORKDIR /fairml-src
COPY requirements.txt /fairml-src

COPY --from=Stage01 /fairml-src/fairml.jar /fairml-src/

RUN apt-get update -y && \
  apt-get autoremove -y
  
RUN apt-get install -y python3-minimal nocache && \
  apt-get autoremove -y
  
RUN apt-get install -y python3-pip nocache && \
  apt-get autoremove -y
  
RUN python3 -m pip install -r requirements.txt --no-cache-dir && \
  rm -rf /fairml-src/requirements.txt

WORKDIR /fairml

ENTRYPOINT ["java", "-cp", "/fairml-src/fairml.jar", "org.eclipse.epsilon.fairml.generator.FairML"]
