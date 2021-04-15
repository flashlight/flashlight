Flashlight and its dependencies can also be built with the provided Dockerfiles. Both CUDA and CPU backends are supported with Docker. The current Docker images are frozen at **Ubuntu 18.04** and **CUDA 10.0**; we update these periodically.

## Docker images on [Docker Hub](https://hub.docker.com/r/flml/flashlight/tags)

Docker images for the CUDA and CPU backends for each Flashlight commit are [available on Docker Hub](https://hub.docker.com/r/flml/flashlight/tags).

### Running Flashlight with Docker

- Install [Docker](https://docs.docker.com/engine/installation)
- If using the CUDA backend, install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)
- Run the given Dockerfile in a new container:
```shell
# if using the CUDA backend
sudo docker run --runtime=nvidia --rm -itd --ipc=host --name flashlight flml/flashlight:cuda-latest
# if using the CPU backend
sudo docker run --rm -itd --name flashlight flml/flashlight:cpu-latest
# start a terminal session inside the container
sudo docker exec -it flashlight bash
```

To run tests inside a container containing an already-built version of Flashlight:
```shell
cd /root/flashlight/build && make test
```

### Building Docker Images from Source

Using the Dockerfiles in this directory:
```shell
git clone --recursive https://github.com/flashlight/flashlight.git
cd flashlight
# for CUDA backend
sudo docker build -f .docker/Dockerfile-CUDA -t flashlight .
# for CPU backend
sudo docker build -f .docker/Dockerfile-CPU -t flashlight .
```

## Logging inside Docker Containers
To ensure logs are displayed, using the `--logtostderr=1` and `--minloglevel=0` flags is best-practice.
