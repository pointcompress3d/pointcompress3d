#pragma once

#include <pcl/range_image/range_image.h>

#include <pcl/pcl_macros.h>
#include <pcl/common/distances.h>
#include <pcl/common/point_tests.h>    // for pcl::isFinite
#include <pcl/common/vector_average.h> // for VectorAverage3f

namespace pcl
{
  class RangeImageWithoutInterpolation : public pcl::RangeImageSpherical
  {
  public:
    // Add two additional attributes in the pcl::RangeImageSpherical class.
    std::vector<float> intensities;
    std::vector<float> azimuth;

    template <typename PointCloudType>
    void
    createFromPointCloudWithoutInterpolation(const PointCloudType &point_cloud, float angular_resolution_x, std::vector<double> elevation_offsets, std::vector<int> az_shifts, int az_increments,
                                             const Eigen::Affine3f &sensor_pose, RangeImage::CoordinateFrame coordinate_frame,
                                             float noise_level, float min_range, float threshold, int border_size)
    /* Project point cloud to range, azimuth and intensity images.
    Modified from the PCL method createFromPointCloud with fixed image size and without interpolation.*/
    {
      // std:: cout << "Start" << std::endl;
      width = point_cloud.height; //1800 or 1812
      height = point_cloud.width; //32
      // Calculate and set the inverse of angular resolution.
      // setAngularResolution(angular_resolution_x, 1.0f);
      setAngularResolution(angular_resolution_x, 1.0f);

      is_dense = false;
      // Get coordinate transformation matrix.
      getCoordinateFrameTransformation(coordinate_frame, to_world_system_);

      to_world_system_ = sensor_pose * to_world_system_;
      to_range_image_system_ = to_world_system_.inverse(Eigen::Isometry);
      unsigned int size = 64 * 2048;
      points.clear();
      points.resize(size, unobserved_point);
      intensities.clear();
      intensities.resize(size, 0);
      azimuth.clear();
      azimuth.resize(size, 0);
      // Calculate and write Z buffer (range/depth), azimuth and intensity into vectors.
      doZBufferWithoutInterpolation(point_cloud, noise_level, min_range, threshold, elevation_offsets, az_shifts, az_increments);
      // std:: cout << "End" << std::endl;
    }

    template <typename PointCloudType>
    void
    doZBufferWithoutInterpolation(const PointCloudType &point_cloud, float noise_level, float min_range, float threshold, std::vector<double> elevation_offsets, std::vector<int> az_shifts, int az_increments)
    {
      using PointType2 = typename PointCloudType::PointType;
      const typename pcl::PointCloud<PointType2>::VectorType &points2 = point_cloud.points;
      unsigned int size = width * height;
      float range_of_current_point, intensity_of_current_point;
      int x, y;
      int num_nan = 0, num_outliers = 0;

      int x_rangeimage, idx_image, num_points = 0;

      // Apply azimuth shift to align pixels in one column with similar azimuth angles.
      for (int i = 0; i < point_cloud.height; i++)
      {
        // point_cloud.width is 32
        for (int j = 0; j < point_cloud.width; j++)
        {
          //get current point
          int idx_cloud = i * height + j;
          // std::cout << "Index:" << idx_cloud << std::endl;
          // Apply azimuth shift.
          x_rangeimage = i + az_shifts[j];
          if (x_rangeimage < 0)
          {
            x_rangeimage += az_increments;
          }
          if (x_rangeimage > az_increments - 1)
          {
            x_rangeimage -= az_increments;
          }

          // Shift front view to the center.
          if (x_rangeimage < 0)
          {
            x_rangeimage += az_increments;
          }
          if (x_rangeimage > az_increments - 1)
          {
            x_rangeimage -= az_increments;
          }

          idx_image = j * 2048 + x_rangeimage;
          // std::cout << i << ":" << j << "=" << x_rangeimage << "|" << idx_image << "<-" << idx_cloud << std::endl;
          float &range_at_image_point = points[idx_image].range;
          float &intensity_at_image_point = intensities[idx_image];
          float &azimuth_at_image_point = azimuth[idx_image];

          //if there is no reflection --> do nothing
          if (!isFinite(points2[idx_cloud]))
          { // Check for NAN etc
            continue;
          }
          else
          {
            // Get vector of the current point.
            Vector3fMapConst current_point = points2[idx_cloud].getVector3fMap();
            // Calculate range of the current point.
            range_of_current_point = current_point.norm();
            // Only store points within min and max range.
            // std::cout << "range:" << range_of_current_point << std::endl;
            if (range_of_current_point < min_range || range_of_current_point > threshold)
            {
              // std::cout << "continue" << std::endl;
              continue;
            }

            // std:: cout << points.size() << "/" << idx_image << std::endl;
            range_at_image_point = range_of_current_point;
            intensity_at_image_point = points2[idx_cloud].intensity*255;
            // Transform to range image coordinate system. Result is [y,z,x].
            Eigen::Vector3f transformedPoint = to_range_image_system_ * current_point;
            // Calculate azimuth angle of current point by arctan(y/x).
            azimuth_at_image_point = atan2LookUp(transformedPoint[0], transformedPoint[2]);
            num_points++;
          }
        }
      }
      // std::cout << "NANs: " << num_nan << std::endl;
      // std::cout << "Num points doing Z buffer:" << num_points << std::endl;
    }

    void
    createEmpty(int cols, int rows, float angular_resolution_x, const Eigen::Affine3f &sensor_pose, RangeImage::CoordinateFrame coordinate_frame)
    // Create empty object filled with NaN points.
    {
      setAngularResolution(angular_resolution_x, 1.0f);
      width = cols;
      height = rows;

      is_dense = false;
      getCoordinateFrameTransformation(coordinate_frame, to_world_system_);
      to_world_system_ = sensor_pose * to_world_system_;
      to_range_image_system_ = to_world_system_.inverse(Eigen::Isometry);
      unsigned int size = width * height;
      points.clear();
      // Initialize as all -inf
      points.resize(size, unobserved_point);
      intensities.clear();
      intensities.resize(size, 0);
      azimuth.clear();
      azimuth.resize(size, 0);
    }

    void
    recalculate3DPointPositionsVelodyne(std::vector<double> &elevation_offsets, float angular_resolution_x, int cols, int rows)
    // Recalculate point cloud from range, azimuth and intensity images.
    {
      int num_points = 0;
      for (int y = 0; y < rows; ++y)
      {
        for (int x = 0; x < cols; ++x)
        {
          PointWithRange &point = points[y * width + x]; //it's from instance that called this class
          if (!std::isinf(point.range))
          {
            float angle_x = azimuth[y * width + x];
            // Elevation angle can be looked up in table elevation_offsets by the row index.
            float angle_y = pcl::deg2rad(elevation_offsets[63 - y]);
            float cosY = std::cos(angle_y);
            // Recalculate point positions.
            Eigen::Vector3f point_tmp = Eigen::Vector3f(point.range * std::sin(-angle_x) * cosY, point.range * std::cos(angle_x) * cosY, point.range * std::sin(angle_y));
            point.x = point_tmp[1];
            point.y = point_tmp[0];
            point.z = point_tmp[2];
            num_points++;
          }
        }
      }
    }
  };

}