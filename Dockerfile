# Offline robot-control pipeline test: no FrankaPy or robot connection is installed.
FROM ros:noetic-ros-core-focal
# ROS core already includes common_msgs. Disable the retired ROS apt source;
# the only additional packages below come from Ubuntu Focal.
RUN rm -f /etc/apt/sources.list.d/ros*.list /etc/apt/sources.list.d/ros*.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3-numpy python3-scipy python3-yaml \
    && rm -rf /var/lib/apt/lists/*
RUN . /opt/ros/noetic/setup.sh && python3 -c "import rospy, geometry_msgs.msg, nav_msgs.msg, sensor_msgs.msg, std_msgs.msg"
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    ROS_MASTER_URI=http://127.0.0.1:11311 \
    ROS_IP=127.0.0.1 \
    ROS_HOSTNAME=127.0.0.1
WORKDIR /app
COPY catchbox/ catchbox/
COPY artifacts/steady-motion-model/actor.npz artifacts/steady-motion-model/final_model.zip artifacts/steady-motion-model/
COPY docker/ docker/
# The ROS image entrypoint sources Noetic. The test creates its own ROS master.
CMD ["python3", "docker/ros_pipeline_test.py"]
