#!/bin/bash          
PROJECT_DIR="./"
DATASET_DIR="/your/path/to/dataset"
sudo docker remove -f geer

echo "mount projects: $PROJECT_DIR --> geer:/home"
echo "mount datasets: $DATASET_DIR --> geer:/media"

sudo docker run --name geer -t -d --gpus 'all,"capabilities=compute,utility,graphics"' --net=host \
    --volume="$DATASET_DIR:/media:rw" \
    --volume="$PROJECT_DIR:/home:rw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY --env=QT_X11_NO_MINTSHM=1 \
    --env=sibr_rg=/workspace/3dgs_dependency/sibr_core/install/bin/SIBR_remoteGaussian_app \
    --env=sibr_gv=/workspace/3dgs_dependency/sibr_core/install/bin/SIBR_gaussianViewer_app \
    --env=baseline=/home/baselines \
    zixunh/3dgeer:latest

# Notes: to add root to or remove root from access control list.
xhost -si:localuser:root
xhost si:localuser:root

sudo docker exec -it -w /home geer /bin/bash --login

