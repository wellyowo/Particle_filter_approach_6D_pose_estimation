#!/usr/bin/env bash

IMG=uwwee/particle-filter-pose-estimator:latest


containerid=$(docker ps -aqf "ancestor=${IMG}") && echo $containerid
xhost +
docker exec -it \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -e LINES="$(tput lines)" \
    ${containerid} \
    bash
xhost -