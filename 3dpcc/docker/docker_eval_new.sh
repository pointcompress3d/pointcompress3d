# in order to be able to use this script install:
# pip install docker-run-cli
DIR="$(cd -P "$(dirname "$0")" && pwd)"
DATASETS=/home/rama/data/a9_dataset_r02_s02
SAMSUNG_4TB=/mnt/ssd_4tb_samsung/

if docker ps -a --format '{{.Names}}' | grep -q "pcl"; then
    echo "Start Existing Container"
    docker start -ai pcl
else
    echo "Start from image"
    docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --gpus all -v $SAMSUNG_4TB:/mnt/ssd_4tb_samsung -v $DATASETS:/data -v "$(dirname "$DIR")/catkin_ws:/catkin_ws" --workdir "/catkin_ws" --name pcl tillbeemelmanns/pointcloud_compression:tf2.11 /bin/bash
    #docker run -it --gpus all -v $SAMSUNG_4TB:/mnt/ssd_4tb_samsung -v $DATASETS:/data -v /home/rama:/home/rama --workdir "/home/rama/rwth-rnn/catkin_ws" --name pcl tillbeemelmanns/pointcloud_compression:tf2.11 /bin/bash
fi

