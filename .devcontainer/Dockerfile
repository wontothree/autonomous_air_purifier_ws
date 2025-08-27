# # Docker Base Image
# # https://hub.docker.com/layers/osrf/ros/humble-desktop-full/images/sha256-71ae08a6a0aae71a2f981e066c8a1d7dd76e956abf419c04626a0c746c3ebf4f
# FROM osrf/ros:humble-desktop-full

# # Make Work Space Directory
# RUN mkdir -p /root/autonomous_navigation_ws/src
# WORKDIR /root/autonomous_navigation_ws/src

# # Install Dependencies

# Docker Base Image
FROM osrf/ros:humble-desktop-full

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
# ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV ALSOFT_DRIVERS=null

# Create workspace
RUN mkdir -p /root/autonomous_navigation_ws/src
WORKDIR /root/autonomous_navigation_ws

# Update & install minimal packages for X11 forwarding
RUN apt-get update && \
    apt-get install -y \
        x11-apps \
        x11-utils \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        mesa-utils \
        xauth \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xfixes0 \
        && rm -rf /var/lib/apt/lists/*

# Copy src if needed
# COPY src /root/autonomous_navigation_ws/src

# Source ROS setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
