docker stop fairml
docker rm fairml
docker rmi fairml
docker system prune
docker build -t fairml .
