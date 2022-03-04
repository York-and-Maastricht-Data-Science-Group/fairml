# run the following command to use this docker container later after installing
# docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -w demo.flexmi
# docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml demo.flexmi
# docker run --rm -d -i -t -v %cd%:/fairml --hostname=fairml -p 8888:8888 --name=fairml fairml -j demo.ipynb
# docker exec fairml bash -c "jupyter notebook list"
FROM ubuntu:20.04

WORKDIR /fairml-src
COPY requirements.txt /fairml-src

RUN apt-get update && \
  apt-get install -y openjdk-11-jre-headless=11.0.13+8-0ubuntu1~20.04 && \
  apt-get install -y git=1:2.25.1-1ubuntu3.2 && \
  apt-get install -y maven=3.6.3-1 && \
  apt-get install -y python3-minimal=3.8.2-0ubuntu2 && \
  apt-get install -y python3-tk=3.8.10-0ubuntu1~20.04 && \ 
  apt-get install -y python3-pip=20.0.2-5ubuntu1.6 && \
  python3 -m pip install -r requirements.txt && \
  git clone -b v0.0.4 https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git && \
  cd /fairml-src/fairml/generator/org.eclipse.epsilon.fairml.generator

WORKDIR /fairml-src/fairml/generator/org.eclipse.epsilon.fairml.generator

RUN mvn install && \
  mkdir /fairml

WORKDIR /fairml

ENTRYPOINT ["java", "-cp", "/fairml-src/fairml/generator/org.eclipse.epsilon.fairml.generator/fairml.jar", "org.eclipse.epsilon.fairml.generator.FairML"]
#ENTRYPOINT ["fairml"]
