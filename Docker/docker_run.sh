# #!/usr/bin/env bash

ARGS=("$@")

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
PROJ_NAME="Particle_filter_approach_6D_pose_estimation"
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
   else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')
# DOCKER_OPTS=""
DOCKER_OPTS=
BASH_OPTION=bash
DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')
if dpkg --compare-versions 19.03 gt "$DOCKER_VER"; then
    echo "Docker version is less than 19.03, using nvidia-docker2 runtime"
    if ! dpkg --list | grep nvidia-docker2; then
        echo "Please either update docker-ce to a version greater than 19.03 or install nvidia-docker2"
        exit 1
    fi
    DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
else
    DOCKER_OPTS="$DOCKER_OPTS --gpus all"
fi

#Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
    echo "[$XAUTH] was not properly created. Exiting..."
   exit 1
fi

docker run \
    -it \
    --rm \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "$XAUTH:$XAUTH" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "/home/$USER/$PROJ_NAME:/home/arg/$PROJ_NAME" \
    -w "/home/arg/$PROJ_NAME" \
    --user "root:root" \
    --name Particle_filter_approach_6D_pose_estimation \
    --network host \
    --privileged \
    --security-opt seccomp=unconfined \
    $DOCKER_OPTS \
    uwwee/particle-filter-pose-estimator:latest \
    $BASH_OPTION
# docker run $DOCKER_OPTS -it \
#     -v /home/$USER/Particle_filter_approach_6D_pose_estimation:/home/wellyuwe/Particle_filter_approach_6D_pose_estimation \
#     -w "/home/wellyuwe/Particle_filter_approach_6D_pose_estimation" \
#     -v "/tmp/.X11-unix:/tmp/.X11-unix" \
#     -e DISPLAY \
#     -v "/etc/localtime:/etc/localtime:ro" \
#     -v "/dev:/dev" \
#     -v "/var/run/docker.sock:/var/run/docker.sock" \
#     --name Particle_filter_approach_6D_pose_estimation \
#     --user "root:root" \
#     --network host \
#     --rm \
#     --privileged \
#     --security-opt seccomp=unconfined \
#     uwwee/particle-filter-pose-estimator:latest \
#     bash