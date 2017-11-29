//
// Created by konrad on 11/8/17.
//

#include "pcl/filters/boost.h"
#include <pcl/filters/filter.h>
#include <sixriver/gpu/filters/gpu_min_max_3d.h>
#include <Eigen/Core>
#define HAVE_CUDA
#ifdef HAVE_CUDA
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#endif
#include <iostream>
#define DEBUG(x) do { std::cerr <<  __FILE__ << " " << \
                                __LINE__ << " " << __func__ << " " << x << std::endl; } while(0);

namespace sixriver {
    namespace gpu {
        namespace filters {
#ifndef HAVE_CUDA

            constexpr bool getMinMax3D(const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                                       Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt) {
                return false;
            }

            constexpr bool getMinMax3D(const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                                       const std::string &distance_field_name, float min_distance, float max_distance,
                                       Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt, bool limit_negative = false) {
                return false;
            }

#else

            struct gpu_4f_point {
                //float data[4];
                float x;
                float y;
                float z;
                float d;

                __host__ __device__ gpu_4f_point() {
                    //data[0] = data[1] = data[2] = data[3] = 0;
                    x = y = z = d = 0;
                }

                __host__ __device__ gpu_4f_point(const gpu_4f_point &other) {
                    //for (int i = 0; i != 4; ++i) {
                    //    data[i] = other.data[i];
                    //}
                    x = other.x;
                    y = other.y;
                    z = other.z;
                    d = other.d;
                }

                __host__ __device__ gpu_4f_point & operator=(const gpu_4f_point &other) {
                    if (&other == this) {
                        return *this;
                    }
                    //for (int i = 0; i != 4; ++i) {
                    //    data[i] = other.data[i];
                    //}
                    x = other.x;
                    y = other.y;
                    z = other.z;
                    d = other.d;
                    return *this;
                }

                __host__ __device__ float & operator[](int index) {
                    return x;//data[index];
                }

                __host__ __device__ float operator[](int index) const {
                    return x;//data[index];
                }
                __host__ __device__ void setConstant(float max) {
                    //data[0] = data[1] = data[2] = data[3] = max;
                    x = y = z = d = max;
                }

                __host__ __device__ void min(const gpu_4f_point & other) {
                    //for(int i = 0; i!=4; ++i) {
                    //    data[i] = data[i] < other.data[i] ? data[i] : other.data[i];
                    //}
                }

                __host__ __device__ void max(const gpu_4f_point & other) {
                    //for(int i = 0; i!=4; ++i) {
                    //    data[i] = data[i] > other.data[i] ? data[i] : other.data[i];
                    //}
                }
            };


            template<typename T>
            struct convert_indefinite_to_min_value : public thrust::unary_function<T,T>
            {
                __host__ __device__ T operator()(const T &x) const
                {
                    if (!isfinite (x[0]) ||
                               !isfinite (x[1]) ||
                               !isfinite (x[2])) {
                        T result;
                        result.setConstant (-FLT_MAX);
                        return result;
                    } else {
                        return x;
                    }
                }
            };

            template<typename T>
            struct convert_indefinite_to_max_value : public thrust::unary_function<T,T>
            {
                __host__ __device__ T operator()(const T &x) const
                {
                    if (!isfinite (x[0]) ||
                        !isfinite (x[1]) ||
                        !isfinite (x[2])) {
                        T result;
                        result.setConstant (FLT_MAX);
                        return result;
                    } else {
                        return x;
                    }
                }
            };

            template<typename T>
            class convert_indefinite_to_min_value_with_limits : public thrust::unary_function<T,T>
            {
                float max_limit_;
                float min_limit_;
                bool limit_negative_;
                int distance_index_;
                const int x_index_;
                const int y_index_;
                const int z_index_;

            public:
                convert_indefinite_to_min_value_with_limits(float min_limit, float max_limit, bool limit_negative, int distance_index, int x_index, int y_index, int z_index) : thrust::unary_function<T, T>(), min_limit_(min_limit), max_limit_(max_limit), limit_negative_(limit_negative), distance_index_(distance_index), x_index_(x_index), y_index_(y_index), z_index_(z_index)
                {
                }

                __host__ __device__ T operator()(const T &x) const
                {
                    // Get the distance value
                    float distance_value = 0;
                    if(distance_index_ == x_index_) {
                        distance_value = x[x_index_];
                    } else if (distance_index_ == y_index_) {
                        distance_value = x[y_index_];
                    } else {
                        distance_value = x[z_index_];
                    }
                    if (limit_negative_)
                    {
                        // Use a threshold for cutting out points which inside the interval
                        if ((distance_value < max_limit_) && (distance_value > min_limit_))
                        {
                            T result;
                            result.setConstant (-FLT_MAX);
                            return result;
                        }
                    }
                    else
                    {
                        // Use a threshold for cutting out points which are too close/far away
                        if ((distance_value > max_limit_) || (distance_value < min_limit_))
                        {
                            T result;
                            result.setConstant (-FLT_MAX);
                            return result;
                        }
                    }


                    if (!isfinite (x[0]) ||
                        !isfinite (x[1]) ||
                        !isfinite (x[2])) {
                        T result;
                        result.setConstant (-FLT_MAX);
                        return result;
                    } else {
                        return x;
                    }
                }
            };

            template<typename T>
            class convert_indefinite_to_max_value_with_limits : public thrust::unary_function<T,T>
            {
                float max_limit_;
                float min_limit_;
                bool limit_negative_;
                int distance_index_;
                int x_index_;
                int y_index_;
                int z_index_;

            public:
                convert_indefinite_to_max_value_with_limits(float min_limit, float max_limit, bool limit_negative, int distance_index, int x_index, int y_index, int z_index) : thrust::unary_function<T, T>(), min_limit_(min_limit), max_limit_(max_limit), limit_negative_(limit_negative), distance_index_(distance_index), x_index_(x_index), y_index_(y_index), z_index_(z_index) {
                }
                __host__ __device__ T operator()(const T &x) const
                {
                    // Get the distance value
                    float distance_value = 0;
                    if(distance_index_ == x_index_) {
                        distance_value = x[x_index_];
                    } else if (distance_index_ == y_index_) {
                        distance_value = x[y_index_];
                    } else {
                        distance_value = x[z_index_];
                    }
                    if (limit_negative_)
                    {
                        // Use a threshold for cutting out points which inside the interval
                        if ((distance_value < max_limit_) && (distance_value > min_limit_))
                        {
                            T result;
                            result.setConstant (FLT_MAX);
                            return result;
                        }
                    }
                    else
                    {
                        // Use a threshold for cutting out points which are too close/far away
                        if ((distance_value > max_limit_) || (distance_value < min_limit_))
                        {
                            T result;
                            result.setConstant (FLT_MAX);
                            return result;
                        }
                    }


                    if (!isfinite (x[0]) ||
                        !isfinite (x[1]) ||
                        !isfinite (x[2])) {
                        T result;
                        result.setConstant (FLT_MAX);
                        return result;
                    } else {
                        return x;
                    }
                }
            };

            template<typename T>
            struct compute_minimum_of_bounding_box : public thrust::binary_function<T,T,T>
            {
                __host__ __device__ T operator()(const T &x, const T&y) const
                {
                    T result = x;
                    result.min(y);
                    return result;
                }
            };

            template<typename T>
            struct compute_maximum_of_bounding_box : public thrust::binary_function<T,T,T>
            {
                __host__ __device__ T operator()(const T &x, const T&y) const
                {
                    T result = x;
                    result.max(y);
                    return result;
                }
            };

            __global__ void convertToDeviceVector(unsigned char * data, int nr_points, int xyz_offset0, int xyz_offset1, int xyz_offset2, int point_step, gpu_4f_point* all_distance_points) {
                int point = blockIdx.x * blockDim.x + threadIdx.x;
                if (point >= nr_points) {
                    return;
                }

                int point_address = point * point_step;
                gpu_4f_point pt;
                memcpy (&pt[0], &data[point_address + xyz_offset0], sizeof (float));
                memcpy (&pt[1], &data[point_address + xyz_offset1], sizeof (float));
                memcpy (&pt[2], &data[point_address + xyz_offset2], sizeof (float));

                all_distance_points[point] = pt;
            }


            __host__ bool getMinMax3D (const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                              Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt) {
                // @todo fix this
                if (cloud->fields[x_idx].datatype != pcl::PCLPointField::FLOAT32 ||
                    cloud->fields[y_idx].datatype != pcl::PCLPointField::FLOAT32 ||
                    cloud->fields[z_idx].datatype != pcl::PCLPointField::FLOAT32)
                {
                    PCL_ERROR ("[pcl::getMinMax3D] XYZ dimensions are not float type!\n");
                    return false;
                }

                gpu_4f_point min_p, max_p;
                min_p.setConstant (FLT_MAX);
                max_p.setConstant (-FLT_MAX);

                size_t nr_points = cloud->width * cloud->height;

                Eigen::Array4f pt = Eigen::Array4f::Zero ();
                /*
                cuInit(0);
                CUcontext context;
                CUdevice device;
                auto result = cuDeviceGet ( &device, 0 );
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to get device");
                }
                result = cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST,device);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to create device context");
                }
                result = cuCtxPushCurrent(context);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to push context");
                }
                CUstream stream;
                result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to create device");
                }
                 */
                cudaStream_t stream;
                cudaStreamCreate(&stream);//, cudaStreamNonBlocking);
                uint8_t * gpu_data_cloud;
                auto runtimeResult = cudaMalloc(&gpu_data_cloud, sizeof(cloud->data[0]) * cloud->data.size());
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to allocate storage");
                }
                runtimeResult = cudaMemcpy(gpu_data_cloud, cloud->data.data(), sizeof(cloud->data[0]) * cloud->data.size(), cudaMemcpyHostToDevice );
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to cp[u data to device");
                }
                //cuMemHostRegister(reinterpret_cast<void *>(const_cast<uint8_t *>(cloud->data.data())), sizeof(cloud->data[0]) * cloud->data.size(), CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
                //cuMemHostGetDevicePointer();

                //CUdeviceptr device_ptr;
                //cuMemHostGetDevicePointer(&device_ptr, reinterpret_cast<void *>(const_cast<uint8_t *>(cloud->data.data())), 0);
                //cuMemcpy();
                DEBUG("initializing all_distance_points");
                thrust::device_vector<gpu_4f_point> all_distance_points(nr_points);
                DEBUG("initialized all_all_distance_points");

                DEBUG("Launching the kernel");
                dim3 threadsPerBlock(1024, 1);
                dim3 numBlocks(static_cast<int> (std::ceil(static_cast<float>(cloud->data.size()) / threadsPerBlock.x)),
                               1);

                //kernel launch

                //convertToDeviceVector<<<threadsPerBlock, numBlocks, 0, stream>>>((unsigned char *)gpu_data_cloud, nr_points, cloud->fields[x_idx].offset, cloud->fields[y_idx].offset, cloud->fields[z_idx].offset, cloud->point_step, thrust::raw_pointer_cast(&all_distance_points[0]));
                DEBUG("KERNEL completed");
                convert_indefinite_to_max_value<gpu_4f_point> f1;
                compute_minimum_of_bounding_box<gpu_4f_point> f2;
                convert_indefinite_to_min_value<gpu_4f_point> g1;
                compute_maximum_of_bounding_box<gpu_4f_point> g2;

                DEBUG("launching thrust reduce operations");
                auto min_value = thrust::transform_reduce(thrust::cuda::par.on(stream), all_distance_points.begin(), all_distance_points.end(), f1, min_p, f2);
                auto max_value = thrust::transform_reduce(thrust::cuda::par.on(stream), all_distance_points.begin(), all_distance_points.end(), g1, max_p, g2);

                DEBUG("done thrust reduce operations");

                runtimeResult = cudaStreamSynchronize(stream);
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to synchronize  stream");
                }
                runtimeResult = cudaFree(gpu_data_cloud);
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to free data");
                }
                runtimeResult = cudaStreamDestroy(stream);
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to destroy stream");
                }
                //cuMemHostUnregister(reinterpret_cast<void *>(const_cast<uint8_t *>(cloud->data.data())));
                /*
                result = cuStreamSynchronize(stream);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to synchronize  stream");
                }
                runtimeResult = cudaFree(gpu_data_cloud);
                if (CUDA_SUCCESS!=runtimeResult) {
                    DEBUG("failed to free data");
                }
                result = cuStreamDestroy(stream);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to destroy stream");
                }
                result = cuCtxSynchronize();
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to synchronize context");
                }
                result = cuCtxPopCurrent(&context);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to pop context");
                }
                result = cuCtxDestroy(context);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to destroy context");
                }*/

                for (int i = 0; i!=4; ++i) {
                    min_pt[i] = min_value[i];
                    max_pt[i] = max_value[i];
                }
                return true;
            }

            bool getMinMax3D (const pcl::PCLPointCloud2ConstPtr &cloud, int x_idx, int y_idx, int z_idx,
                         const std::string &distance_field_name, float min_distance, float max_distance,
                         Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt, bool limit_negative) {
        // @todo fix this
          if (cloud->fields[x_idx].datatype != pcl::PCLPointField::FLOAT32 ||
              cloud->fields[y_idx].datatype != pcl::PCLPointField::FLOAT32 ||
              cloud->fields[z_idx].datatype != pcl::PCLPointField::FLOAT32)
          {
            PCL_ERROR ("[pcl::getMinMax3D] XYZ dimensions are not float type!\n");
            return false;
          }
                gpu_4f_point min_p, max_p;

          min_p.setConstant (FLT_MAX);
          max_p.setConstant (-FLT_MAX);

          // Get the distance field index
          int distance_idx = pcl::getFieldIndex (*cloud, distance_field_name);

          // @todo fix this
          if (cloud->fields[distance_idx].datatype != pcl::PCLPointField::FLOAT32)
          {
            PCL_ERROR ("[pcl::getMinMax3D] Filtering dimensions is not float type!\n");
            return false;
          }

          size_t nr_points = cloud->width * cloud->height;

          Eigen::Array4f pt = Eigen::Array4f::Zero ();


                /*
                cuInit(0);
                CUcontext context;
                CUdevice device;
                auto result = cuDeviceGet ( &device, 0 );
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to get device");
                }
                result = cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST,device);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to create device context");
                }
                result = cuCtxPushCurrent(context);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to push context");
                }
                CUstream stream;
                result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to create device");
                }
                 */
                cudaStream_t stream;
                cudaStreamCreate(&stream);//, cudaStreamNonBlocking);
                thrust::device_vector<gpu_4f_point> all_distance_points(nr_points);

                uint8_t * gpu_data_cloud;
                auto runtimeResult = cudaMalloc(&gpu_data_cloud, sizeof(cloud->data[0]) * cloud->data.size());
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to allocate storage");
                }
                runtimeResult = cudaMemcpy(gpu_data_cloud, cloud->data.data(), sizeof(cloud->data[0]) * cloud->data.size(), cudaMemcpyHostToDevice );
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to cp[u data to device");
                }

                //cuMemHostRegister(reinterpret_cast<void *>(const_cast<uint8_t *>(cloud->data.data())), sizeof(cloud->data[0]) * cloud->data.size(), CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
                //cuMemHostGetDevicePointer();

                //CUdeviceptr device_ptr;
                //cuMemHostGetDevicePointer(&device_ptr, reinterpret_cast<void *>(const_cast<uint8_t *>(cloud->data.data())), 0);
                //cuMemcpy();
                dim3 threadsPerBlock(1024, 1);
                dim3 numBlocks(static_cast<int> (std::ceil(static_cast<float>(cloud->data.size()) / threadsPerBlock.x)),
                               1);

                //kernel launch

                convertToDeviceVector<<<threadsPerBlock, numBlocks, 0, stream>>>((unsigned char *)gpu_data_cloud, nr_points, cloud->fields[x_idx].offset, cloud->fields[y_idx].offset, cloud->fields[z_idx].offset, cloud->point_step, thrust::raw_pointer_cast(&all_distance_points[0]));

                convert_indefinite_to_max_value_with_limits<gpu_4f_point> f1(min_distance,max_distance,limit_negative,distance_idx, x_idx,y_idx,z_idx);
                compute_minimum_of_bounding_box<gpu_4f_point> f2;
                convert_indefinite_to_min_value_with_limits<gpu_4f_point> g1(min_distance,max_distance,limit_negative,distance_idx, x_idx,y_idx,z_idx);
                compute_maximum_of_bounding_box<gpu_4f_point> g2;

                auto min_value = thrust::transform_reduce(thrust::cuda::par.on(stream), all_distance_points.begin(), all_distance_points.end(), f1, min_p, f2);
                auto max_value = thrust::transform_reduce(thrust::cuda::par.on(stream), all_distance_points.begin(), all_distance_points.end(), g1, max_p, g2);


                runtimeResult = cudaStreamSynchronize(stream);
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to synchronize  stream");
                }
                runtimeResult = cudaFree(gpu_data_cloud);
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to free data");
                }
                runtimeResult = cudaStreamDestroy(stream);
                if (cudaSuccess!=runtimeResult) {
                    DEBUG("failed to destroy stream");
                }
                //cuMemHostUnregister(reinterpret_cast<void *>(const_cast<uint8_t *>(cloud->data.data())));
                /*
                result = cuStreamSynchronize(stream);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to synchronize  stream");
                }
                runtimeResult = cudaFree(gpu_data_cloud);
                if (CUDA_SUCCESS!=runtimeResult) {
                    DEBUG("failed to free data");
                }
                result = cuStreamDestroy(stream);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to destroy stream");
                }
                result = cuCtxSynchronize();
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to synchronize context");
                }
                result = cuCtxPopCurrent(&context);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to pop context");
                }
                result = cuCtxDestroy(context);
                if (CUDA_SUCCESS!=result) {
                    DEBUG("failed to destroy context");
                }
                 */
                for (int i = 0; i!=4; ++i) {
                    min_pt[i] = min_value[i];
                    max_pt[i] = max_value[i];
                }
                return true;

            }
#endif
        }
    }
}