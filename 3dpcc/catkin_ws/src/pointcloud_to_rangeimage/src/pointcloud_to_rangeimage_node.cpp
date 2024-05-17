#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <image_transport/image_transport.h>

#include <dynamic_reconfigure/server.h>
#include <pointcloud_to_rangeimage/PointCloudToRangeImageReconfigureConfig.h>

#include <pcl/point_types.h>
#include <pcl/range_image/range_image_spherical.h>
#include <velodyne_pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>

#include <boost/thread/mutex.hpp>

#include "pointcloud_to_rangeimage/utils.h"
#include "pointcloud_to_rangeimage/RangeImage.h"
#include "pointcloud_to_rangeimage/range_image_expand.h"
#include <chrono>


namespace fs = boost::filesystem;

namespace
{
  typedef velodyne_pcl::PointXYZIRT PointType;
  typedef pcl::PointCloud<PointType> PointCloud;

  typedef pcl::RangeImage RI;
  typedef pcl::RangeImageWithoutInterpolation RIS;

  typedef pointcloud_to_rangeimage::PointCloudToRangeImageReconfigureConfig conf;
  typedef dynamic_reconfigure::Server<conf> RangeImageReconfServer;

  typedef image_transport::ImageTransport It;
  typedef image_transport::Publisher Pub;
}
class RangeImageConverter
/*
Receive messages of type sensor_msgs::PointCloud2 from topic /velodyne_points, project the point clouds to range,
azimuth and intensity images. The images are packed in message of type RangeImage and published to the topic /msg_out.
*/
{
private:
  bool _compressed = false;
  bool _newmsg;
  bool _laser_frame;
  bool _record_images;
  std::string _record_path;

  // RangeImage frame
  pcl::RangeImage::CoordinateFrame _frame;

  // RangeImage resolution
  double _ang_res_x;
  double _azi_scale = (2 * static_cast<float>(M_PI)) / std::numeric_limits<ushort>::max();

  // Sensor min/max range
  float _min_range;
  float _max_range;

  double _vlp_rpm;
  double _firing_cycle;
  int _az_increments;
  double _threshold;
  std::vector<int> _az_shifts;
  std::vector<double> _azimuth_offsets;
  std::vector<double> _elevation_offsets;
  boost::mutex _mut;

  cv::Mat _rangeImage;
  cv::Mat _intensityMap;
  cv::Mat _azimuthMap;
  PointCloud _pointcloud;

  boost::shared_ptr<RIS> rangeImageSph_;

  ros::NodeHandle nh_;
  It it_r_;
  It it_i_;
  Pub pub_r_;
  Pub pub_i_;

  ros::Publisher pub_;
  ros::Subscriber sub_;

  boost::shared_ptr<RangeImageReconfServer> drsv_;
  pointcloud_to_rangeimage::RangeImage riwi_msg;

public:
  RangeImageConverter() : _newmsg(false),
                          _laser_frame(true),
                          _ang_res_x(600 * (1.0 / 60.0) * 360.0 * 0.000055296),
                          _min_range(0.5),
                          _max_range(200),
                          it_r_(nh_),
                          it_i_(nh_),
                          nh_("~")
  {
    // Get parameters from configuration file.
    while (!nh_.getParam("/point_cloud_to_rangeimage/vlp_rpm", _vlp_rpm))
    {
      ROS_WARN("Could not get Parameter 'vlp_rpm'! Retrying!");
    }
    ROS_INFO_STREAM("RPM set to: " << _vlp_rpm);

    while (!nh_.getParam("/point_cloud_to_rangeimage/firing_cycle", _firing_cycle))
    {
      ROS_WARN("Could not get Parameter 'firing_cycle'! Retrying!");
    }
    ROS_INFO_STREAM("Firing Cycle set to: " << _firing_cycle << " s");

    while (!nh_.getParam("/point_cloud_to_rangeimage/elevation_offsets", _elevation_offsets))
    {
      ROS_WARN("Could not get Parameter 'elevation_offsets'! Retrying!");
    }

    while (!nh_.getParam("/point_cloud_to_rangeimage/azimuth_offsets", _azimuth_offsets))
    {
      ROS_WARN("Could not get Parameter 'azimuth_offsets'! Retrying!");
    }

    while (!nh_.getParam("/point_cloud_to_rangeimage/record_images", _record_images))
    {
      ROS_WARN("Could not get Parameter 'record_images'! Retrying!");
    }

    while (!nh_.getParam("/point_cloud_to_rangeimage/record_path", _record_path))
    {
      ROS_WARN("Could not get Parameter 'record_path'! Retrying!");
    }

    while (!nh_.getParam("/point_cloud_to_rangeimage/threshold", _threshold))
    {
      ROS_WARN("Could not get Parameter 'threshold'! Retrying!");
    }

    // std::reverse(_elevation_offsets.begin(), _elevation_offsets.end());
    std::reverse(_azimuth_offsets.begin(), _azimuth_offsets.end());

    // Calculate angular resolution.
    _ang_res_x = _vlp_rpm * (1.0 / 60.0) * 360.0 * _firing_cycle;

    // Calculate azimuth shifts in pixel.
    _az_shifts.resize(64);
    for (int i = 0; i < 64; i++)
    {
      _az_shifts[i] = (int)round(_azimuth_offsets[i] / _ang_res_x);
    }

    
    rangeImageSph_ = boost::shared_ptr<RIS>(new RIS);
    drsv_.reset(new RangeImageReconfServer(ros::NodeHandle("pointcloud_to_rangeimage_dynreconf")));

    RangeImageReconfServer::CallbackType cb;
    cb = boost::bind(&RangeImageConverter::drcb, this, _1, _2);

    drsv_->setCallback(cb);

    nh_.param("laser_frame", _laser_frame, _laser_frame);

    double min_range = static_cast<double>(_min_range);
    double max_range = static_cast<double>(_max_range);
    double threshold = static_cast<double>(_threshold);

    nh_.param("min_range", min_range, min_range);
    nh_.param("max_range", max_range, max_range);

    _min_range = static_cast<float>(min_range);
    _max_range = static_cast<float>(max_range);

    pub_r_ = it_r_.advertise("image_out", 1);
    pub_i_ = it_i_.advertise("intensity_map", 1);
    pub_ = nh_.advertise<pointcloud_to_rangeimage::RangeImage>("msg_out", 1);
    ros::NodeHandle nh;
    sub_ = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, &RangeImageConverter::callback, this);

    // Using laser frame by default.
    _frame = (_laser_frame) ? pcl::RangeImage::LASER_FRAME : pcl::RangeImage::CAMERA_FRAME;
  }

  ~RangeImageConverter()
  {
  }

  void callback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    if (msg == NULL)
      return;
    boost::mutex::scoped_lock(_mut);

    // Set header for the RangeImage message.
    pcl_conversions::toPCL(msg->header, _pointcloud.header);
    pcl::fromROSMsg(*msg, _pointcloud); //after this width=32, height=1800
    // std::string filename = "/data/temp/s110_lidar_ouster_south/1698077884_571100650_s110_lidar_ouster_south.pcd";

    //FIELDS x y z intensity ring time --> theirs
    //FIELDS x y z intensity --> ours
    riwi_msg.header = msg->header;
    // Begin of processing for evaluating application latency.
    riwi_msg.send_time = ros::Time::now();

    _newmsg = true;
  }

  void dump_azimuth_to_file(std::vector<float> azimuth, int rows, int cols, const std::string& filename) {
      std::ofstream outputFile(filename);
      if (!outputFile.is_open()) {
          std::cerr << "Error: Unable to open file: " << filename << std::endl;
          return;
      }

      for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              outputFile << azimuth[i * cols + j] << " ";
          }
          outputFile << std::endl;
      }

      outputFile.close();
  }

  std::vector<PointType, Eigen::aligned_allocator<PointType>> order(std::vector<PointType, Eigen::aligned_allocator<PointType>>& rows) {
      int width = 2048;
      int height = 64;
      int start = rows.size() - width + 1;
      std::vector<PointType, Eigen::aligned_allocator<PointType>> new_rows;

      for (int i = start; i < start + width; ++i) {
          for (int hop = i - 1; hop >= 0; hop -= width) {
              // Add the element at index hop to new_rows
              new_rows.push_back(rows[hop]);
          }
      }

      return new_rows;
  }

  int convert(int index, std::vector<std::string>* file_list, std::vector<double>* times)
  {
    // What the point if nobody cares ?
    // if (pub_.getNumSubscribers() <= 0)
    //   return;

    // if (!_newmsg || index >= file_list->size())
    //   return index;

    std::cout << index << ":" << file_list->at(index) << std::endl;
    if (pcl::io::loadPCDFile<PointType>(file_list->at(index), _pointcloud) == -1) {
        // Error handling if the file cannot be loaded
        PCL_ERROR("Couldn't read file pointcloud.pcd\n");
        exit(1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    _pointcloud.points = order(_pointcloud.points);

    _pointcloud.width = 64;
    _pointcloud.height = 2048;
    _pointcloud.header.stamp = index;

    boost::mutex::scoped_lock(_mut);

    //fixed image size
    int cols = 2048; // int cols = 1812;
    int rows = 64; // int rows = 32;
    int height_cloud = _pointcloud.height;
    int width_cloud = _pointcloud.width;


    // Azimuth increment for shifting the rows. See header range_image_expand.h.
    // _az_increments = 1812;
    _az_increments = 2048;
    /* Project point cloud to range, azimuth and intensity imagfes.
    Modified from the PCL method createFromPointCloud with fixed image size and without interpolation.*/
    Eigen::Matrix4f sensor_pose;
    sensor_pose.setIdentity();

    rangeImageSph_->createFromPointCloudWithoutInterpolation(_pointcloud, pcl::deg2rad(_ang_res_x), _elevation_offsets,
                                                             _az_shifts, _az_increments, Eigen::Affine3f(sensor_pose),
                                                             _frame, 0.00, 0.0f, _threshold, 0);


    rangeImageSph_->header.frame_id = _pointcloud.header.frame_id;
    rangeImageSph_->header.stamp = _pointcloud.header.stamp;

    // To convert range to 16-bit integers.
    float factor = 1.0f / (_max_range - _min_range);
    float offset = -_min_range;

    _rangeImage = cv::Mat::zeros(rows, cols, cv_bridge::getCvType("mono16"));
    _intensityMap = cv::Mat::zeros(rows, cols, cv_bridge::getCvType("mono8"));
    _azimuthMap = cv::Mat::zeros(rows, cols, cv_bridge::getCvType("mono16"));
    float r, range, a, azi;
    int reversed_j, num_points = 0;

    // dump azimuth
    // std::string azimuthDump = _record_path + "azimuth_dump/" + fs::path(file_list->at(index)).stem().string() + ".txt";
    // dump_azimuth_to_file(rangeImageSph_->azimuth, rows, cols, azimuthDump);
    // Store the images as OpenCV image.
    for (int i = 0; i < cols; i++)
    {
      for (int j = 0; j < rows; j++) //32
      {
        reversed_j = rows - 1 - j;
        r = rangeImageSph_->points[reversed_j * cols + i].range;
        a = rangeImageSph_->azimuth[reversed_j * cols + i];
        azi = (a + static_cast<float>(M_PI)) / _azi_scale;
        range = factor * (r + offset);
        _rangeImage.at<ushort>(j, i) = static_cast<ushort>((range)*std::numeric_limits<ushort>::max());
        _intensityMap.at<uchar>(j, i) = static_cast<uchar>(rangeImageSph_->intensities[reversed_j * cols + i]);
        _azimuthMap.at<ushort>(j, i) = static_cast<ushort>(azi);
        if (range != 0.0f)
          num_points++;
      }
    }

    // Fill the nans in the images with previous pixel value.
    riwi_msg.NansRow.clear();
    riwi_msg.NansCol.clear();
    int num_nan = 0;
    for (int i = 0; i < cols; i++)
    {
      for (int j = 0; j < rows; ++j) //32
      {
        if (_rangeImage.at<ushort>(j, i) == 0)
        {
          int i_pre = (i != 0) ? i - 1 : 2047;
          // _rangeImage.at<ushort>(j, i) = _rangeImage.at<ushort>(j, i_pre);
          // _intensityMap.at<uchar>(j, i) = _intensityMap.at<uchar>(j, i_pre);
          // Can't fill left azi with right value, bacause only azi is not continuous.
          _azimuthMap.at<ushort>(j, i) = (i != 0) ? _azimuthMap.at<ushort>(j, i - 1) : 0;
          riwi_msg.NansRow.push_back(static_cast<uchar>(j));
          riwi_msg.NansCol.push_back(static_cast<ushort>(i));
          num_nan++;
        }
      }
    }

    riwi_msg.RangeImage = *(cv_bridge::CvImage(std_msgs::Header(), "mono16", _rangeImage).toImageMsg());
    riwi_msg.IntensityMap = *(cv_bridge::CvImage(std_msgs::Header(), "mono8", _intensityMap).toImageMsg());
    riwi_msg.AzimuthMap = *(cv_bridge::CvImage(std_msgs::Header(), "mono16", _azimuthMap).toImageMsg());

    auto end = std::chrono::high_resolution_clock::now();
    auto ms_int =  std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    times->push_back(ms_int.count());
    
    // Create image dataset from rosbag.
    if (_record_images)
    {
      std::stringstream ss;
      ss << fs::path(file_list->at(index)).stem().string();
      std::string azimuthName = _record_path + "azimuth/" + ss.str() + ".png";
      std::string intensityName = _record_path + "intensity/" + ss.str() + ".png";
      std::string rangeName = _record_path + "range/" + ss.str() + ".png";
      cv::imwrite(intensityName, _intensityMap);
      cv::imwrite(rangeName, _rangeImage);
      cv::imwrite(azimuthName, _azimuthMap);
      std::cout << "Write:" << rangeName << std::endl;
      // write pcd file
      std::string pcdName = _record_path + "pcd/pcd_" + ss.str() + ".pcd";
      pcl::io::savePCDFileASCII(pcdName, _pointcloud);


      if(_compressed)
      {
        azimuthName = _record_path + "azimuth_compressed/" + ss.str() + ".png";
        intensityName = _record_path + "intensity_compressed/" + ss.str() + ".png";
        rangeName = _record_path + "range_compressed/" + ss.str() + ".png";
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9); // Quality level (0-100), higher means better quality but larger file size
        cv::imwrite(rangeName, _rangeImage, compression_params);
        cv::imwrite(azimuthName, _azimuthMap, compression_params);
        cv::imwrite(intensityName, _intensityMap, compression_params);
      }
    }

    pub_r_.publish(riwi_msg.RangeImage);
    pub_i_.publish(riwi_msg.IntensityMap);
    pub_.publish(riwi_msg);

    _newmsg = false;
    return ++index;
  }

private:
  void drcb(conf &config, uint32_t level)
  {
    _min_range = config.min_range;
    _max_range = config.max_range;
    _laser_frame = config.laser_frame;

    _frame = (_laser_frame) ? pcl::RangeImage::LASER_FRAME : pcl::RangeImage::CAMERA_FRAME;

    ROS_INFO_STREAM("ang_res_x " << _ang_res_x);
    ROS_INFO_STREAM("min_range " << _min_range);
    ROS_INFO_STREAM("max_range " << _max_range);

    if (_laser_frame)
      ROS_INFO_STREAM("Frame type : "
                      << "LASER");
    else
      ROS_INFO_STREAM("Frame type : "
                      << "CAMERA");
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_to_rangeimage");

  RangeImageConverter converter;

  ros::Rate rate(15);

  std::string directory = "/data/temp/s110_lidar_ouster_south/";
  std::vector<std::string> file_list;

  fs::path dir_path(directory);

  if (fs::exists(dir_path) && fs::is_directory(dir_path)) {
      for (fs::directory_iterator itr(dir_path); itr != fs::directory_iterator(); ++itr) {
          if (fs::is_regular_file(itr->status())) {
              file_list.push_back(itr->path().string());
          }
      }
  } else {
   std::cerr << "Directory not found or not a directory: " << directory << std::endl;
      return 1;
  }

  std::sort(file_list.begin(), file_list.end());
  
  int index = 0;
  std::vector<double> times;
  while (ros::ok() && index < file_list.size())
  {
    index = converter.convert(index, &file_list, &times);
    ros::spinOnce();

    rate.sleep();
  }


  double duration = 0;
  for(auto t:times){
    duration += t;
  }


  double average_time = duration / file_list.size();
  std::cout << "Avg time:" << average_time << std::endl;
}
