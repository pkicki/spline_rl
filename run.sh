docker stop spline_rl
docker rm spline_rl
xhost + local:root

docker run -it \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --volume="$(pwd)/spline_rl:/home/user/spline_rl" \
    --env PYTHONPATH="/home/user/spline_rl:/usr/local/lib/python3.8/dist-packages" \
    --privileged \
    --network=host \
    --name=spline_rl \
    spline_rl
