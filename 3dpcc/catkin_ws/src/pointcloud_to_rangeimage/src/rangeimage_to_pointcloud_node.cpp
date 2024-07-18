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
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <boost/thread/mutex.hpp>

#include "pointcloud_to_rangeimage/utils.h"
#include <pointcloud_to_rangeimage/RangeImage.h>
#include "pointcloud_to_rangeimage/range_image_expand.h"

namespace fs = boost::filesystem;

std::string base_path = "/catkin_ws/images";
std::string azimuth_path = base_path + "/azimuth";
std::string intensity_path = base_path + "/intensity";
std::string range_path = base_path + "/range";
std::vector<std::string> ranges, azis, intensities;

namespace
{
  typedef pcl::PointXYZI PointType;
  typedef pcl::PointCloud<PointType> PointCloud;

  typedef pcl::RangeImage RI;
  typedef pcl::RangeImageWithoutInterpolation RIS;

  typedef pointcloud_to_rangeimage::PointCloudToRangeImageReconfigureConfig conf;
  typedef dynamic_reconfigure::Server<conf> RangeImageReconfServer;

  typedef image_transport::ImageTransport It;
  typedef image_transport::Subscriber Sub;
}

class PointCloudConverter
/*
Receive messages of type RangeImage from topic /msg_decoded, project the range, azimuth and intensity images
back to point cloud. The point cloud of type sensor_msgs::PointCloud2 is then published to the topic /pointcloud_out.
*/
{
private:
  bool _newmsg;
  bool _laser_frame;
  bool _init;

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
  std::vector<int> _az_shifts;
  std::vector<double> _azimuth_offsets;
  std::vector<double> _elevation_offsets;
  std::vector<int> _nans_row;
  std::vector<int> _nans_col;

  boost::mutex _mut;

  std_msgs::Header _header;
  ros::Time _send_time;

  cv_bridge::CvImagePtr _rangeImage;
  cv_bridge::CvImagePtr _intensityMap;
  cv_bridge::CvImagePtr _azimuthMap;

  PointCloud _pointcloud;
  boost::shared_ptr<RIS> rangeImageSph_;
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
  It it_;
  float delay_sum_=0;
  int frame_count_=0;
  boost::shared_ptr<RangeImageReconfServer> drsv_;

public:
  PointCloudConverter() : _newmsg(false),
                          _laser_frame(true),
                          _init(false),
                          _ang_res_x(600 * (1.0 / 60.0) * 360.0 * 0.000055296),
                          _min_range(0.5),
                          _max_range(200),
                          it_(nh_),
                          nh_("~")
  {
    rangeImageSph_ = boost::shared_ptr<RIS>(new RIS);
    drsv_.reset(new RangeImageReconfServer(ros::NodeHandle("rangeimage_to_pointcloud_dynreconf")));
    RangeImageReconfServer::CallbackType cb;
    cb = boost::bind(&PointCloudConverter::drcb, this, _1, _2);
    drsv_->setCallback(cb);

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

    std::reverse(_elevation_offsets.begin(), _elevation_offsets.end());
    nh_.param("laser_frame", _laser_frame, _laser_frame);

    double min_range = static_cast<double>(_min_range);
    double max_range = static_cast<double>(_max_range);
    nh_.param("min_range", min_range, min_range);
    nh_.param("max_range", max_range, max_range);
    _min_range = static_cast<float>(min_range);
    _max_range = static_cast<float>(max_range);

    _ang_res_x = _vlp_rpm * (1.0 / 60.0) * 360.0 * _firing_cycle;
    _az_increments = (int)ceil(360.0f / _ang_res_x); //1809
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_out", 1);

    std::string transport = "raw";
    nh_.param("transport", transport, transport);
    if (transport != "raw" && transport != "compressedDepth")
    {
      ROS_WARN_STREAM("Transport " << transport
                                   << ".\nThe only transports supported are :\n\t - raw\n\t - compressedDepth.\n"
                                   << "Setting transport default 'raw'.");
      transport = "raw";
    }
    else
      ROS_INFO_STREAM("Transport " << transport);
    image_transport::TransportHints transportHint(transport);
    std::string image_in = "image_in";
    nh_.param("image_in", image_in, image_in);

    sub_ = nh_.subscribe("/msg_decoded", 1, &PointCloudConverter::callback, this);
    _frame = (_laser_frame) ? pcl::RangeImage::LASER_FRAME : pcl::RangeImage::CAMERA_FRAME;
  }

  ~PointCloudConverter()
  {
  }

  void callback(const pointcloud_to_rangeimage::RangeImageConstPtr &msg)
  {
    if (msg == NULL)
      return;

    boost::mutex::scoped_lock lock(_mut);

    // Copy images to OpenCV image pointer.
    try
    {
      _rangeImage = cv_bridge::toCvCopy(msg->RangeImage, msg->RangeImage.encoding);
      _intensityMap = cv_bridge::toCvCopy(msg->IntensityMap, msg->IntensityMap.encoding);
      _azimuthMap = cv_bridge::toCvCopy(msg->AzimuthMap, msg->AzimuthMap.encoding);
      _header = msg->header;
      _send_time = msg->send_time;
      _newmsg = true;
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_WARN_STREAM(e.what());
    }

    // Get nan coordinates
    _nans_row.clear();
    _nans_col.clear();
    for (int i = 0; i < msg->NansRow.size(); i++)
    {
      _nans_row.push_back(static_cast<int>(msg->NansRow[i]));
      _nans_col.push_back(static_cast<int>(msg->NansCol[i]));
    }

//    cv::imwrite("/home/rosuser/catkin_ws/images/rangeimage_intensity_decoded.png", _intensityMap->image);
//    cv::imwrite("/home/rosuser/catkin_ws/images/rangeimage_range_decoded.png", _rangeImage->image);
//    cv::imwrite("/home/rosuser/catkin_ws/images/rangeimage_azimuth_decoded.png", _azimuthMap->image);
  }

  int convert(int index, std::vector<double>* times)
  {
    // What the point if nobody cares ?
    //if (pub_.getNumSubscribers() <= 0)
    //  return;
    // if (_rangeImage == NULL)
    //   return index;
    // if (!_newmsg)
    //   return index;
    if(index >= ranges.size())
      return index;
    _pointcloud.clear();

    
    // std::string filename = filenames->at(index);
    // cv::Mat _rangeImage = cv::imread(range_path + "/" + filename + ".png");
    // cv::Mat _azimuthMap = cv::imread(azimuth_path + "/" + filename + ".png");
    // cv::Mat _intensityMap = cv::imread(intensity_path + "/" + filename + ".png");

    cv::Mat _rangeImage = cv::imread(range_path + "/" + ranges.at(index)+".jpg");
    cv::Mat _azimuthMap = cv::imread(azimuth_path + "/" + azis.at(index)+".png");
    cv::Mat _intensityMap = cv::imread(intensity_path + "/" + intensities.at(index)+".png");


    // // Check if the images were loaded successfully
    // if (r.empty() || a.empty() || i.empty()) {
    //     std::cerr << "Error: Could not open or find one of the images." << std::endl;
    //     return index;
    // }
    
    auto start = std::chrono::high_resolution_clock::now();
    float factor = 1.0f / (_max_range - _min_range);
    float offset = -_min_range;

    _mut.lock();

    int cols = _rangeImage.cols;
    int rows = _rangeImage.rows;

    rangeImageSph_->createEmpty(cols, rows, pcl::deg2rad(_ang_res_x), Eigen::Affine3f::Identity(), _frame);
    bool mask[64][2048]{false};
    
    // Mask nans as -1 in the images.
    for (int i = 0; i < _nans_row.size(); i++)
    {
      mask[_nans_row[i]][_nans_col[i]] = true;
    }

    // Decode ranges.
    int point_counter = 0;
    int uninf_counter = 0;
    for (int i = 0; i < cols; ++i)
    {
      for (int j = 0; j < rows; ++j)
      {
        ushort range_img = _rangeImage.at<ushort>(j, i);
        // Discard unobserved points.
        if (mask[j][i] || range_img == 0)
          continue;

        // Rescale range. 0-1 range_img /65535
        float range = static_cast<float>(range_img) /
                      static_cast<float>(std::numeric_limits<ushort>::max());

        range = (range - offset * factor) / factor; //0.33 / (1/200) => distance in meter
        pcl::PointWithRange &p = rangeImageSph_->getPointNoCheck(i, j);
        // Read intensity value.
        float &intensity = rangeImageSph_->intensities[j * cols + i];
        intensity = static_cast<float>(_intensityMap.at<uchar>(j, i));
        // Read azimuth angle.
        ushort azi_img = _azimuthMap.at<ushort>(j, i);
        float azi = static_cast<float>(azi_img) * _azi_scale - static_cast<float>(M_PI);
        float &azimuth = rangeImageSph_->azimuth[j * cols + i];
        azimuth = static_cast<float>(azi); //copy the azi to rangeImageSph_->azimuth
        p.range = range;
        point_counter++;
      }
    }

    // Reconstruct 3D point positions.
    rangeImageSph_->recalculate3DPointPositionsVelodyne(_elevation_offsets, pcl::deg2rad(_ang_res_x), cols, rows);

    // Reconstruct point cloud.
    for (int i = 0; i < rangeImageSph_->points.size(); ++i)
    {
      pcl::PointWithRange &pts = rangeImageSph_->points[i];
      // Discard unobserved points
      if (std::isinf(pts.range))
        continue;

      // PointType p(pts.x, pts.y, pts.z);
      PointType p;
      p.x = pts.x;
      p.y = pts.y;
      p.z = pts.z;
      p.intensity = rangeImageSph_->intensities[i];

      _pointcloud.push_back(p);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms_int =  std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    times->push_back(ms_int.count());

    // Dumped point cloud to file
    // std::string pcd_filename = base_path + "/pcd_reconstructed/" + azis.at(index) + ".pcd";
    // std::cout << "Dumping:" << pcd_filename << std::endl;
    // if(!_pointcloud.empty()){ 
    //   pcl::io::savePCDFileASCII(pcd_filename, _pointcloud);
    // }
    
    //clear points
    _init = false;

    sensor_msgs::PointCloud2 ros_pc;
    pcl::toROSMsg(_pointcloud, ros_pc);
    ros_pc.header = _header;
    delay_sum_ += ros::Time::now().toSec() - _send_time.toSec();
    frame_count_++;
    // Calculate delay of application.
    // std::cout << "Delay: " << ros::Time::now().toSec() - _send_time.toSec() << " seconds" << std::endl;
    // std::cout << "Average delay: " << delay_sum_/frame_count_ << " seconds" << std::endl;
    pub_.publish(ros_pc);
    _newmsg = false;
    _mut.unlock();
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

    _init = false;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rangeimage_to_pointcloud");
  
  PointCloudConverter converter;

  ros::Rate rate(15);

  int index = 0;
  // std::vector<std::string> ranges, azis, intensities;

  if (fs::exists(range_path) && fs::is_directory(range_path)) {
      for (fs::directory_iterator itr(range_path); itr != fs::directory_iterator(); ++itr) {
          if (fs::is_regular_file(itr->status())) {
              ranges.push_back(itr->path().stem().string());
          }
      }

      for(fs::directory_iterator itr(intensity_path); itr!=fs::directory_iterator(); ++itr){
          if (fs::is_regular_file(itr->status())) {
              azis.push_back(itr->path().stem().string());
          }
      }

      for(fs::directory_iterator itr(azimuth_path); itr!=fs::directory_iterator(); ++itr){
          if (fs::is_regular_file(itr->status())) {
              intensities.push_back(itr->path().stem().string());
          }
      }
  } else {
      std::cerr << "Directory not found or not a directory: " << range_path << std::endl;
      return 1;
  }

  std::sort(ranges.begin(),ranges.end());
  std::sort(azis.begin(),azis.end());
  std::sort(intensities.begin(), intensities.end());
  std::vector<double> times;

  while (
    ros::ok() 
    && index < std::min({ranges.size(), azis.size(), intensities.size()})
  ){
    index = converter.convert(index, &times);

    ros::spinOnce();

    rate.sleep();
  }


  double duration = 0;
  for(auto t:times){
    duration += t;
  }


  double average_time = duration / ranges.size();
  std::cout << "Avg time:" << average_time << std::endl;
}
