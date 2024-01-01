# #!/usr/bin/env bash

ARGS=("$@")

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
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

DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')
DOCKER_OPTS=""

#Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
    echo "[$XAUTH] was not properly created. Exiting..."
   exit 1
fi

docker run $DOCKER_OPTS -it \
    -v /home/$USER/Particle_filter_approach_6D_pose_estimation:/home/wellyuwe/Particle_filter_approach_6D_pose_estimation \
    -w "/home/wellyuwe/Particle_filter_approach_6D_pose_estimation" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -e DISPLAY \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    --name Particle_filter_approach_6D_pose_estimation \
    --user "root:root" \
    --network host \
    --rm \
    --privileged \
    --security-opt seccomp=unconfined \
    uwwee/particle-filter-pose-estimator:latest \
    bash