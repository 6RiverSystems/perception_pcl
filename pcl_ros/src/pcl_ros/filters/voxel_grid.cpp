/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: voxel_grid.cpp 35876 2011-02-09 01:04:36Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include "pcl_ros/filters/voxel_grid.h"
#include <chrono>

//////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl_ros::VoxelGrid::child_init (ros::NodeHandle &nh, bool &has_service)
{
  // Enable the dynamic reconfigure service
  has_service = true;
  srv_ = boost::make_shared <dynamic_reconfigure::Server<pcl_ros::VoxelGridConfig> > (nh);
  dynamic_reconfigure::Server<pcl_ros::VoxelGridConfig>::CallbackType f = boost::bind (&VoxelGrid::config_callback, this, _1, _2);
  srv_->setCallback (f);

  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::VoxelGrid::filter (const PointCloud2::ConstPtr &input, 
                            const IndicesPtr &indices, 
                            PointCloud2 &output)
{
  auto start = std::chrono::system_clock::now();

  boost::mutex::scoped_lock lock (mutex_);
  auto lock_end_time = std::chrono::system_clock::now();

  pcl::PCLPointCloud2::Ptr pcl_input(new pcl::PCLPointCloud2);
  auto pcl_input_created = std::chrono::system_clock::now();

  pcl_conversions::toPCL (*(input), *(pcl_input));
  auto pcl_converted = std::chrono::system_clock::now();

  impl_.setInputCloud (pcl_input);
  auto impl_input_set = std::chrono::system_clock::now();

  impl_.setIndices (indices);
  auto impl_indices_set = std::chrono::system_clock::now();

  pcl::PCLPointCloud2 pcl_output;
  impl_.filter (pcl_output);
  auto filter_applied = std::chrono::system_clock::now();

  pcl_conversions::moveFromPCL(pcl_output, output);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> total_time = end - start;
  std::chrono::duration<double> lock_time = lock_end_time - start;
  std::chrono::duration<double> input_creation_time = pcl_input_created - lock_end_time;
  std::chrono::duration<double> input_conversion_time = pcl_converted - pcl_input_created;
  std::chrono::duration<double> impl_input_set_time = impl_input_set - pcl_converted;
  std::chrono::duration<double> indices_time = impl_indices_set - impl_input_set;
  std::chrono::duration<double> filter_time = filter_applied - impl_indices_set;
  std::chrono::duration<double> output_conversion_time = end - filter_applied;


  ROS_INFO_THROTTLE(10, "voxel_grid processing took: %f seconds. lock acquisition took: %f seconds"
                   ", input creation took: %f seconds, input conversion took %f seconds, "
  "setting input took %f seconds, indices took %f seconds, filter took %f seconds, output conversion took %f seconds",
  total_time.count(), lock_time.count(), input_conversion_time.count(), input_conversion_time.count(),
  impl_input_set_time.count(), indices_time.count(), filter_time.count(), output_conversion_time.count());
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::VoxelGrid::config_callback (pcl_ros::VoxelGridConfig &config, uint32_t level)
{
  boost::mutex::scoped_lock lock (mutex_);

  Eigen::Vector3f leaf_size;


  if (config.leaf_size_x > 0 && config.leaf_size_y > 0 && config.leaf_size_z > 0)
  {
    NODELET_WARN("pconfig_callback] All leaf values are set. Using the leaf_size_x, leaf_size_y, leaf_size_z values.");
    leaf_size.setConstant (0);
    leaf_size[0] = config.leaf_size_x;
    leaf_size[1] = config.leaf_size_y;
    leaf_size[2] = config.leaf_size_z;
  }
  else
  {
    NODELET_ERROR("[config_callback] completely unexpected condition happened. Values are: leaf_size_x = %f, leaf_size_y = %f, leaf_size_z = %f. Setting voxel grid size to 0.01.",  config.leaf_size_x, config.leaf_size_y, config.leaf_size_z);
    leaf_size = {0.01, 0.01, 0.01};
  }
  if (leaf_size != impl_.getLeafSize())
  {
    NODELET_DEBUG ("[config_callback] Setting the downsampling leaf size to: %f, %f, %f.", leaf_size[0], leaf_size[1], leaf_size[2]);
    impl_.setLeafSize(leaf_size[0], leaf_size[1], leaf_size[2]);
  }

  double filter_min, filter_max;
  impl_.getFilterLimits (filter_min, filter_max);
  if (filter_min != config.filter_limit_min)
  {
    filter_min = config.filter_limit_min;
    NODELET_DEBUG ("[config_callback] Setting the minimum filtering value a point will be considered from to: %f.", filter_min);
  }
  if (filter_max != config.filter_limit_max)
  {
    filter_max = config.filter_limit_max;
    NODELET_DEBUG ("[config_callback] Setting the maximum filtering value a point will be considered from to: %f.", filter_max);
  }
  impl_.setFilterLimits (filter_min, filter_max);

  if (impl_.getFilterLimitsNegative () != config.filter_limit_negative)
  {
    impl_.setFilterLimitsNegative (config.filter_limit_negative);
    NODELET_DEBUG ("[%s::config_callback] Setting the filter negative flag to: %s.", getName ().c_str (), config.filter_limit_negative ? "true" : "false");
  }

  if (impl_.getFilterFieldName () != config.filter_field_name)
  {
    impl_.setFilterFieldName (config.filter_field_name);
    NODELET_DEBUG ("[config_callback] Setting the filter field name to: %s.", config.filter_field_name.c_str ());
  }

  // ---[ These really shouldn't be here, and as soon as dynamic_reconfigure improves, we'll remove them and inherit from Filter
  if (tf_input_frame_ != config.input_frame)
  {
    tf_input_frame_ = config.input_frame;
    NODELET_DEBUG ("[config_callback] Setting the input TF frame to: %s.", tf_input_frame_.c_str ());
  }
  if (tf_output_frame_ != config.output_frame)
  {
    tf_output_frame_ = config.output_frame;
    NODELET_DEBUG ("[config_callback] Setting the output TF frame to: %s.", tf_output_frame_.c_str ());
  }
  // ]---
}

typedef pcl_ros::VoxelGrid VoxelGrid;
PLUGINLIB_EXPORT_CLASS(VoxelGrid,nodelet::Nodelet);

