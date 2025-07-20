HostD=/home/user/Documents/Projects/ail-mk2/ && \
ContainerD=/home/user/Documents/Projects/ail-mk2/ && \
docker run --gpus all -it \
-v "${HostD}":"${ContainerD}" \
-w "${ContainerD}" \
--cpuset-cpus 0-11 \
--shm-size=16g \
--name ailmk2 \
ailmk2_image bash
