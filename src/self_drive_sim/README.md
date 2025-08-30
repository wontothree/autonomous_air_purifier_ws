# [Package] self_drive_sim

    self_drive_sim
    ├── launch/
    │   ├── a1_single.rviz           # Rviz 설정 파일
    │   ├── debug_launch_map0.py     # 맵 0 키보드 조종 런치 - 후술할 "키보드 조종" 참고
    │   ├── debug_launch_map1.py     # 맵 1 키보드 조종 런치
    │   ├── debug_launch_map2.py     # 맵 2 키보드 조종 런치
    │   ├── debug_launch_map3.py     # 맵 3 키보드 조종 런치
    │   ├── test_launch_all.py       # test_launch_map 4개를 모두 실행
    │   ├── test_launch_map0.py      # 맵 0 테스트 런치 - 채점 포함
    │   ├── test_launch_map1.py      # 맵 1 테스트 런치
    │   ├── test_launch_map2.py      # 맵 2 테스트 런치
    │   ├── test_launch_map3.py      # 맵 3 테스트 런치
    │   ├── train_launch_map0.py     # 맵 0 훈련 런치 - 채점 미포함, agent.learn을 호출
    │   ├── train_launch_map1.py     # 맵 1 훈련 런치
    │   ├── train_launch_map2.py     # 맵 2 훈련 런치
    │   └── train_launch_map3.py     # 맵 3 훈련 런치
    │
    ├── resources
    ├── self_drive_sim
    │   ├── agent
    │   │   ├── agent.py             # 자율주행 에이전트 관련: 실제 인공지능 구현은 모두 이 디렉토리 내에서 이루어집니다
    │   │   └── interfaces.py        # Observation 등 시뮬레이션-에이전트 간 데이터 교환 포맷 정의
    │   │
    │   ├── simulation               # 시뮬레이션 동작 관련
    │   │   ├── floor_map.py         # 매핑 정보 및 MapInfo 정의 코드
    │   │   ├── gazebo_env.py        # Gazebo 통신 및 시뮬레이션 핵심 로직 코드
    │   │   └── pollution_manager.py # 공기 오염 시뮬레이션 코드
    │   │
    │   ├── actor_collision.py       # 동적 오브젝트 작동 코드
    │   ├── debug.py                 # 키보드 조종 사이클 실행 코드
    │   ├── test.py                  # 테스트 사이클 실행 코드
    │   └── train.py                 # 훈련 사이클 실행 코드
    │
    ├── test
    └── worlds
        ├── map0.json
        ├── map0.npz
        ├── map0.world
        ├── map1.json
        ├── map1.npz
        ├── map1.world
        ├── map2.json
        ├── map2.npz
        ├── map2.world
        ├── map3.json
        ├── map3.npz
        └── map3.world

# 🚀 Quick Start

```bash
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
```

```bash
ros2 launch self_drive_sim train_launch_map0.py
ros2 launch self_drive_sim train_launch_map1.py
ros2 launch self_drive_sim train_launch_map2.py
ros2 launch self_drive_sim train_launch_map3.py
```

[http://localhost:8080/vnc.html](http://localhost:8080/vnc.html)

## Teleoperation

```bash
# dependencies
apt-get update
apt-get install -y ros-humble-teleop-twist-keyboard

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/robot_0/cmd_vel
```

# 🗨️ Published Topics

- /clicked_point 
- /clock          
- /goal_pose      
- [geometry_msgs/msg/PoseWithCovarianceStamped](https://docs.ros2.org/foxy/api/geometry_msgs/msg/PoseWithCovarianceStamped.html) (`initialpose`)
- /parameter_events     
- /performance_matrixs  
- /robot_0/base_sensor_camera_sensor/camera_info    
- /robot_0/base_sensor_camera_sensor/image_raw      
- /robot_0/base_sensor_camera_sensor/image_raw/compressed 
- /robot_0/base_sensor_camera_sensor/image_raw/compressedDepth 
- /robot_0/base_sensor_camera_sensor/image_raw/theora          
- /robot_0/base_sensor_lidar_back_controller/out               
- /robot_0/base_sensor_lidar_front_controller/out              
- /robot_0/base_sensor_ray_controller/out                      
- /robot_0/base_sensor_tof_left_sensor/camera_info             
- /robot_0/base_sensor_tof_left_sensor/depth/image_raw         
- /robot_0/base_sensor_tof_left_sensor/depth/image_raw/compressed
- /robot_0/base_sensor_tof_left_sensor/depth/image_raw/compressedDepth
- /robot_0/base_sensor_tof_left_sensor/depth/image_raw/theora
- /robot_0/base_sensor_tof_left_sensor/image_raw               
- /robot_0/base_sensor_tof_left_sensor/image_raw/compressed
- /robot_0/base_sensor_tof_left_sensor/image_raw/compressedDepth
- /robot_0/base_sensor_tof_left_sensor/image_raw/theora
- /robot_0/base_sensor_tof_left_sensor/points                  
- /robot_0/base_sensor_tof_right_sensor/camera_info
- /robot_0/base_sensor_tof_right_sensor/depth/image_raw
- /robot_0/base_sensor_tof_right_sensor/depth/image_raw/compressed
- /robot_0/base_sensor_tof_right_sensor/depth/image_raw/compressedDepth
- /robot_0/base_sensor_tof_right_sensor/depth/image_raw/theora
- /robot_0/base_sensor_tof_right_sensor/image_raw
- /robot_0/base_sensor_tof_right_sensor/image_raw/compressed
- /robot_0/base_sensor_tof_right_sensor/image_raw/compressedDepth
- /robot_0/base_sensor_tof_right_sensor/image_raw/theora
- /robot_0/base_sensor_tof_right_sensor/points
- /robot_0/cmd_vel                    
- /robot_0/collision_raw_0_controller/out
- /robot_0/collision_raw_1_controller/out
- /robot_0/collision_raw_2_controller/out
- /robot_0/collision_raw_3_controller/out
- /robot_0/joint_states
- /robot_0/odom                    
- /robot_0/robot_description       
- /rosout
- /tf                              
- /tf_static                       

# Nodes

`rmc_localizer_node`

# Visualization

```bash
cd ~/autonomous_navigation_ws
source install/setup.bash
cd src/self_drive_sim/self_drive_sim/agent
python3 agent.py
```