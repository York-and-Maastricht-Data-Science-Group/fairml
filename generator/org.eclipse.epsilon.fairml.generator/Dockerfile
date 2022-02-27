# run the following command to use this docker container later after installing
# docker run --rm -i -v %cd%:/data --name=fairml fairml -w demo.flexmi

FROM ubuntu:20.04

WORKDIR /fairml-src

COPY requirements.txt /fairml-src

RUN apt-get update
RUN apt-get install -y git=1:2.25.1-1ubuntu3.2
RUN apt-get install -y openjdk-11-jdk-headless=11.0.13+8-0ubuntu1~20.04 
RUN apt-get install -y maven=3.6.3-1

RUN git clone https://github.com/York-and-Maastricht-Data-Science-Group/fairml.git
RUN cd fairml/generator/org.eclipse.epsilon.fairml.generator
RUN cd mvn install
RUN chmod 777 fairml
ENV PATH "$PATH:/fairml-src/fairml/generator/org.eclipse.epsilon.fairml.generator/"

RUN apt-get install -y python3=3.8.2-0ubuntu2 
RUN apt-get install -y python3-pip=20.0.2-5ubuntu1.6

RUN python3 -m pip install -r requirements.txt

RUN mkdir /fairml
WORKDIR /fairml

#ENTRYPOINT ["java", "-cp", "fairml.jar", "org.eclipse.epsilon.fairml.generator.FairML"]
ENTRYPOINT ["fairml"]
