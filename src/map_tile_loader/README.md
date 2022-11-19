# map_tile_loader

## Dependencies
- dynamic_reconfigure (included but not used yet)
- pcl_conversions
- pcl_ros
- sensor_msgs
- tf2_ros

## Library
### map_loader
- methods:
  - MapLoader
    - input: (optional) string path
  - loadConfig
    - input: string path 
  - setSearchRadius
    - input: double rad
  - getSubmaps
    - input: pcl::PointXYZ center
    - output: pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_ptr

## Nodes
- test_node
  - subscribe: N/A
  - publish: /map (sensor_msgs::PointCloud2)
  
- map_publisher
  - parameters: map_path (String)
  - subscribe: /query_pose (geometry_msgs::PoseStamped)
  - publish: /map (sensor_msgs::PointCloud2)
