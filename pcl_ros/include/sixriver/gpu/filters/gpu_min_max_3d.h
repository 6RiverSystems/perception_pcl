//
// Created by konrad on 11/8/17.
//

#ifndef SIXRIVER_GPU_FILTERS_GPU_MIN_MAX_3D_H
#define SIXRIVER_GPU_FILTERS_GPU_MIN_MAX_3D_H

#include <pcl/filters/boost.h>
#include <pcl/filters/filter.h>

namespace sixriver {
    namespace gpu {
        namespace filters {
#ifndef HAVE_CUDA
            inline bool getMinMax3D (const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                                        Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt) {
                return false;
            }

            inline bool getMinMax3D (const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                                        const std::string &distance_field_name, float min_distance, float max_distance,
                                        Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt, bool limit_negative = false) {
                return false;
            }
#else
            bool getMinMax3D(const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                             Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);

            bool getMinMax3D(const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                             const std::string &distance_field_name, float min_distance, float max_distance,
                             Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt, bool limit_negative = false);
#endif
        }
    }
}
#endif //PCL_GPU_MIN_MAX_3D_H
