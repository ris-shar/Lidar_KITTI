# ------------ base image (ROS 2 Humble) ------------
    FROM ros:humble

    # ------------ environment ------------
    ENV DEBIAN_FRONTEND=noninteractive
    SHELL ["/bin/bash", "-c"]
    
    # ------------ APT packages ------------
    RUN apt-get update && apt-get install -y \
        python3-pip \
        python3-colcon-common-extensions \
        ros-humble-cv-bridge \
        && rm -rf /var/lib/apt/lists/*
    
    # ------------ Python dependencies ------------
    COPY requirements.txt /tmp/
    RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
    
    # ------------ workspace ------------
    WORKDIR /root/kitti_project_ws
    COPY . .
    
    # ------------ build ------------
    RUN source /opt/ros/humble/setup.bash && \
        colcon build --symlink-install
    
    # ------------ default command ------------
    CMD bash -c "source /opt/ros/humble/setup.bash && \
                 source install/setup.bash && \
                 ros2 launch kitti_project kitti_pipeline.launch.py"
    