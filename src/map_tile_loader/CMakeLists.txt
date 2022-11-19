cmake_minimum_required(VERSION 3.0.2)
project(map_tile_loader)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  tf2_ros
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

set(lib_DIR /usr/lib/x84_64-linux-gnu)
link_directories(${lib_DIR})

add_executable(test_node src/test_node.cpp)
target_link_libraries(test_node ${catkin_LIBRARIES} libjsoncpp.a)

add_executable(map_publisher src/map_publisher.cpp)
target_link_libraries(map_publisher ${catkin_LIBRARIES} libjsoncpp.a)
