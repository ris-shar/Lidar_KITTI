cmake_minimum_required(VERSION 3.8)
project(kitti_project)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

#
# install the Python package **without** editable mode
#
ament_python_install_package(${PROJECT_NAME} SKIP_EDITABLE)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
