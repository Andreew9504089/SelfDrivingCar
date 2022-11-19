#include"map_loader.h"

#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/PointCloud2.h>
#include<pcl_conversions/pcl_conversions.h>

sensor_msgs::PointCloud2::Ptr map_cloud(new sensor_msgs::PointCloud2);

ros::Publisher pub_map;

void timerCallback(const ros::TimerEvent& event){
    ROS_INFO("Timer triggered");
    map_cloud->header.stamp = ros::Time::now();
    ROS_INFO("Point cloud size: %d", map_cloud->width);
    pub_map.publish(*map_cloud);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "test_node");
    ros::NodeHandle n("~");

    pub_map = n.advertise<sensor_msgs::PointCloud2>("/map", 1);

    MapLoader<pcl::PointXYZ> loader("/home/biomotion/nuscenes_maps");

    ROS_INFO("Loading submap");
    pcl::PointXYZ point(300., 600., 0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    loader.setSearchRadius(100.);
    if(loader.getSubmaps(point, cloud) < 0){
        ROS_ERROR("Loading submap fail");
    }
    ROS_INFO("Done loading submap");

    pcl::toROSMsg(*cloud, *map_cloud);
    map_cloud->header.frame_id = "map";
    pub_map.publish(*map_cloud);
    ros::Timer timer1 = n.createTimer(ros::Duration(5.), timerCallback);
    ros::spin();
    return 0;
}