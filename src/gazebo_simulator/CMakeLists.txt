cmake_minimum_required(VERSION 3.10)
project(gazebo_simulator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(gazebo_simulator_node src/gazebo_simulator_node.cpp)
target_include_directories(gazebo_simulator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(gazebo_simulator_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS gazebo_simulator_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# add

install(DIRECTORY GazeboFiles
        DESTINATION share/${PROJECT_NAME})
install(DIRECTORY urdf
        DESTINATION share/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.xacro" PATTERN "*.urdf.xacro")

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY 
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY 
  urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY urdf meshes worlds launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.urdf"
  PATTERN "*.xacro"
  PATTERN "*.gazebo"
  PATTERN "*.dae"
  PATTERN "*.world"
  PATTERN "*.launch")

ament_package()
