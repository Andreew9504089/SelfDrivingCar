#ifndef MAP_LOADER_H
#define MAP_LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <ros/ros.h>
#include <jsoncpp/json/json.h>
#include <vector>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

// #define VERBOSE

namespace STATUS{
    int FAIL = -1;
    int GOOD = 0;
    int SAME = 1;
    int NEW = 2;
}

template<typename PointT>
class MapLoader{

    using PointCloud = typename pcl::PointCloud<PointT>;
    using PointCloudPtr = typename pcl::PointCloud<PointT>::Ptr;

    std::string mapPath;
    std::vector<std::string> submapFiles, mapCloudFiles;

    pcl::PointCloud<pcl::PointXYZ>::Ptr centroidCloud;
    pcl::KdTreeFLANN<pcl::PointXYZ> submapKdtree;

    PointCloudPtr mapCloud;

    double searchRad = 50.;
public:
    MapLoader():centroidCloud(new pcl::PointCloud<pcl::PointXYZ>),
                mapCloud(new PointCloud) {}
    MapLoader(const std::string path):centroidCloud(new pcl::PointCloud<pcl::PointXYZ>),
                                      mapCloud(new PointCloud) {
        loadConfig(path);
    }
    int loadConfig(const std::string path){
        mapPath = path;
        return readJSONConfig(path + "/submaps_config.json");
    }
    int readJSONConfig(const std::string filename);
    int readSubmaps(const std::vector<std::string>& files, PointCloudPtr& cloud_ptr);
    void setSearchRadius(double rad) { searchRad = rad; }
    void searchNearbySubmaps(const pcl::PointXYZ center, std::vector<std::string>& foundFiles);
    int getSubmaps(const pcl::PointXYZ center, PointCloudPtr& cloud_ptr);
};

#include "map_loader.hpp"
#endif // MAP_LOADER_H
