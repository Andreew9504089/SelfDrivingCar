#include "map_loader.h"

template <typename PointT>
int MapLoader<PointT>::readJSONConfig(const std::string filename)
{

    std::ifstream ifd(filename);
    if (!ifd.is_open())
    {
        return STATUS::FAIL;
    }

    Json::Reader reader;
    Json::Value root;
    if (reader.parse(ifd, root))
    {
        centroidCloud->width = root["submaps"].size();
        centroidCloud->height = 1;
        centroidCloud->points.resize(centroidCloud->width * centroidCloud->height);
        submapFiles.resize(root["submaps"].size());
        for (unsigned int i = 0; i < root["submaps"].size(); i++)
        {
            submapFiles[i] = root["submaps"][i]["file_name"].asString();

            centroidCloud->points[i].x = root["submaps"][i]["center_x"].asDouble();
            centroidCloud->points[i].y = root["submaps"][i]["center_y"].asDouble();
            centroidCloud->points[i].z = 0.0;
        }
    }
    submapKdtree.setInputCloud(centroidCloud); //build Kd tree
    return STATUS::GOOD;
}

template <typename PointT>
int MapLoader<PointT>::readSubmaps(const std::vector<std::string> &files, PointCloudPtr &cloud_ptr)
{
    PointCloudPtr new_cloud(new PointCloud);
    PointCloudPtr next_cloud(new PointCloud);

    for (std::vector<std::string>::const_iterator it = files.begin(); it != files.end(); it++)
    {
        if (new_cloud->width == 0)
        {
            if (pcl::io::loadPCDFile<PointT>(mapPath + "/" + *it, *new_cloud) == -1)
            {
                return STATUS::FAIL;
            }
        }
        else
        {
            if (pcl::io::loadPCDFile<PointT>(mapPath + "/" + *it, *next_cloud) == -1)
            {
                return STATUS::FAIL;
            }
            new_cloud->insert(new_cloud->end(), next_cloud->begin(), next_cloud->end());
        }
    }
    cloud_ptr = new_cloud;
    return STATUS::GOOD;

}

template<typename PointT>
void MapLoader<PointT>::searchNearbySubmaps(const pcl::PointXYZ center, std::vector<std::string> &foundFiles)
{
    std::vector<int> pointIndices;
    std::vector<float> pointSquaredDistance;
    submapKdtree.radiusSearch(
        center, searchRad, pointIndices, pointSquaredDistance);
    std::sort(pointIndices.begin(), pointIndices.end());
    for (std::vector<int>::iterator idx = pointIndices.begin(); idx != pointIndices.end(); idx++)
    {
        foundFiles.push_back(submapFiles[*idx]);
    }
}

template<typename PointT>
int MapLoader<PointT>::getSubmaps(const pcl::PointXYZ center, PointCloudPtr &cloud_ptr){
    int ret = STATUS::GOOD;
    std::vector<std::string> mapFilesNearCenter;
    searchNearbySubmaps(center, mapFilesNearCenter);
    if(mapCloudFiles != mapFilesNearCenter ){
        mapCloudFiles = mapFilesNearCenter;
#ifdef VERBOSE
        for( auto file : mapCloudFiles){
            std::cout << file << std::endl;
        }
#endif //VERBOSE
        ret = readSubmaps(mapCloudFiles, mapCloud);
        if(ret == STATUS::GOOD){
            ret = STATUS::NEW;
        }
        cloud_ptr = mapCloud;
    }else{
        cloud_ptr = mapCloud;
        ret = STATUS::SAME;
    }
    return ret;
}
