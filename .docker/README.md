Flashlight and its dependencies can also be built with the provided Dockerfile. Both CUDA and CPU backends are supported with Docker. Right now we support **Ubuntu 18.04** and **CUDA 10.0**.

## Docker images on the Docker hub
 
Github actions push new build for both CPU and CUDA backends for each commit on master.
https://hub.docker.com/r/flml/flashlight/tags

## Building image locally
To build Docker image from the source:
  ```sh
  git clone --recursive https://github.com/facebookresearch/flashlight.git
  cd flashlight
  # for CUDA backend
  sudo docker build --no-cache -f .docker/Dockerfile-CUDA -t fl .
  # for CPU backend
  sudo docker build --no-cache -f .docker/Dockerfile-CPU -t fl .
  ```

## Logging in the container
For logging inside a container, use the `--logtostderr=1 --minloglevel=0` flags.