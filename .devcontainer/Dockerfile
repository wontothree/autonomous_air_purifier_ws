# Docker Base Image
FROM osrf/ros:humble-desktop-full

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=0

# Create workspace
RUN mkdir -p /root/autonomous_agent_ws/src
WORKDIR /root/autonomous_agent_ws

# Update & install Gazebo ROS packages
RUN apt-get update && \
    apt-get install -y \
        ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Source ROS setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
