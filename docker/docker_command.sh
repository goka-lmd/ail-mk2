HostD=<path_to_ailmk2> && \
ContainerD=<path_to_ailmk2> && \
docker run --gpus all -it \
-v "${HostD}":"${ContainerD}" \
-w "${ContainerD}" \
--cpuset-cpus 0-11 \
--shm-size=16g \
--name ailmk2 \
ailmk2_image bash
