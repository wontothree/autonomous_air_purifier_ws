cmake_minimum_required(VERSION 3.19)
project(mc_localizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CMake에서 컴파일러가 헤더 파일(.h)을 찾을 경로를 지정한다.
include_directories(include/)

# 의존성을 추가한다.
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)


# src 안에 있는 모든 실행 파일을 추가한다. -> 실행 파일을 만든다.
add_executable(mc_localizer_node 
  src/mc_localizer_node.cpp
  src/mc_localizer_ros.cpp
  src/mc_localizer.cpp
)

# 특정 실행 파일이 의존하는 의존성을 명시한다.
ament_target_dependencies(mc_localizer_node
  rclcpp
  sensor_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  visualization_msgs
)

# 실행 파일을 설치 경로에 복사한다.
install(TARGETS mc_localizer_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()