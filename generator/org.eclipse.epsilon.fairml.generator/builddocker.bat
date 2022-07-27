docker stop fairml-jupyter
docker rm fairml
docker rmi fairml
docker system prune
docker build -t fairml .
