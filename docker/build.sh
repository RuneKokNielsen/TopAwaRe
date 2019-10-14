
docker build --build-arg GIT_USERNAME=$1 --build-arg GIT_PASSWORD=$2 --build-arg CACHEBUST=$(date +%s) --tag=topaware .
