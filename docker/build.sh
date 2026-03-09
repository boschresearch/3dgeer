#!/bin/bash
echo "CUDA_ARCH $1"

rm -rf docker/Dockerfile
cp docker/Dockerfile_$1 docker/Dockerfile
docker -D build -t zixunh/3dgeer:latest -f docker/Dockerfile .