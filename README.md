# movie-classifier

## Build docker image

```
docker image build -t pylearn .
```

## Run container

```
docker run -it -p 8888:8888 -v $pwd'':/home/jovyan/work --name movie-classifier-app pylearn
```
