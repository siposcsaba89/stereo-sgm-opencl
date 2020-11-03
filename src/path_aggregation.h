/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once
//#include ""
#include "device_buffer.hpp"
#include "libsgm_ocl/types.h"
#include "device_kernel.h"
#include <regex>
#include <vector>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ocl_sgm);

#include <opencv2/opencv.hpp>

namespace sgm
{
namespace cl
{

template <int DIRECTION, unsigned int MAX_DISPARITY>
struct VerticalPathAggregation
{
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;
    static constexpr unsigned int DP_BLOCK_SIZE = 16u;

    VerticalPathAggregation(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx)
        , m_cl_device(device)
    {
    }
    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_kernel = nullptr;
    void enqueue_aggregate_up2down_path(DeviceBuffer<cost_type> & dest,
        const DeviceBuffer<feature_type> & left,
        const DeviceBuffer<feature_type> & right,
        int width,
        int height,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream)
    {
        if (!m_kernel)
            init();

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &dest.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(p1), &p1);
        err = clSetKernelArg(m_kernel, 6, sizeof(p2), &p2);
        err = clSetKernelArg(m_kernel, 7, sizeof(min_disp), &min_disp);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

        //setup kernels
        const size_t gdim = (width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
        const size_t bdim = BLOCK_SIZE;
        //
        size_t global_size[1] = { gdim * bdim };
        size_t local_size[1] = { bdim };
        err = clEnqueueNDRangeKernel(stream,
            m_kernel,
            1,
            nullptr,
            global_size,
            local_size,
            0, nullptr, nullptr);
        CHECK_OCL_ERROR(err, "Error finishing queue");
//        clFinish(stream);
//        cv::Mat debug(height, width, CV_8UC4);
//        clEnqueueReadBuffer(stream, dest.data(), true, 0, width * height * 4, debug.data, 0, nullptr, nullptr);
//        cv::imshow("vertical path aggregation debug", debug);
//        cv::waitKey(0);
    }

    void init();

};


template <int DIRECTION, unsigned int MAX_DISPARITY>
struct HorizontalPathAggregation
{
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int DP_BLOCK_SIZE = 8u;
    static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;
    static constexpr unsigned int WARPS_PER_BLOCK = 4u;
    static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;


    HorizontalPathAggregation(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx)
        , m_cl_device(device)
    {
    }
    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_kernel = nullptr;
    void enqueue(DeviceBuffer<cost_type>& dest,
        const DeviceBuffer<feature_type>& left,
        const DeviceBuffer<feature_type>& right,
        int width,
        int height,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream)
    {
        if (!m_kernel)
            init();

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &dest.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(p1), &p1);
        err = clSetKernelArg(m_kernel, 6, sizeof(p2), &p2);
        err = clSetKernelArg(m_kernel, 7, sizeof(min_disp), &min_disp);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        static const unsigned int PATHS_PER_BLOCK =
            BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

        //setup kernels
        const size_t gdim = (height + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
        const size_t bdim = BLOCK_SIZE;
        //
        size_t global_size[1] = { gdim * bdim };
        size_t local_size[1] = { bdim };
        err = clEnqueueNDRangeKernel(stream,
            m_kernel,
            1,
            nullptr,
            global_size,
            local_size,
            0, nullptr, nullptr);
        //cl_int errr = clFinish(stream);
        //CHECK_OCL_ERROR(err, "Error finishing queue");
        //cv::Mat debug(height, width, CV_8UC4);
        //clEnqueueReadBuffer(stream, dest.data(), true, 0, width * height * 4, debug.data, 0, nullptr, nullptr);
        //cv::imshow("horizontal path aggregation debug", debug);
        //cv::waitKey(0);
    }

    void init();

};


template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
struct ObliquePathAggregation
{
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int DP_BLOCK_SIZE = 16u;
    static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

    ObliquePathAggregation(cl_context ctx, cl_device_id device)
        : m_cl_ctx(ctx)
        , m_cl_device(device)
    {
    }
    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_kernel = nullptr;
    void enqueue(DeviceBuffer<cost_type>& dest,
        const DeviceBuffer<feature_type>& left,
        const DeviceBuffer<feature_type>& right,
        int width,
        int height,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream)
    {
        if (!m_kernel)
            init();

        cl_int err;
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &dest.data());
        err = clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &left.data());
        err = clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &right.data());
        err = clSetKernelArg(m_kernel, 3, sizeof(width), &width);
        err = clSetKernelArg(m_kernel, 4, sizeof(height), &height);
        err = clSetKernelArg(m_kernel, 5, sizeof(p1), &p1);
        err = clSetKernelArg(m_kernel, 6, sizeof(p2), &p2);
        err = clSetKernelArg(m_kernel, 7, sizeof(min_disp), &min_disp);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

        const unsigned gdim = (width + height + PATHS_PER_BLOCK - 2) / PATHS_PER_BLOCK;
        const unsigned bdim = BLOCK_SIZE;
        //
        size_t global_size[1] = { gdim * bdim };
        size_t local_size[1] = { bdim };
        err = clEnqueueNDRangeKernel(stream,
            m_kernel,
            1,
            nullptr,
            global_size,
            local_size,
            0, nullptr, nullptr);
        //cl_int errr = clFinish(stream);
        //CHECK_OCL_ERROR(err, "Error finishing queue");
        //cv::Mat debug(height, width, CV_8UC4);
        //clEnqueueReadBuffer(stream, dest.data(), true, 0, width * height * 4, debug.data, 0, nullptr, nullptr);
        //cv::imshow("horizontal path aggregation debug", debug);
        //cv::waitKey(0);
    }

    void init();

};



template <size_t MAX_DISPARITY>
class PathAggregation 
{

private:
    static const unsigned int MAX_NUM_PATHS = 8;

    DeviceBuffer<cost_type> m_cost_buffer;
    std::vector<DeviceBuffer<cost_type>> m_sub_buffers;
    cl_command_queue m_streams[MAX_NUM_PATHS];

public:
    PathAggregation(cl_context ctx,
        cl_device_id device);
    ~PathAggregation();

    const DeviceBuffer<cost_type>& get_output() const;

    void enqueue(
        const DeviceBuffer<feature_type> & left,
        const DeviceBuffer<feature_type>& right,
        int width,
        int height,
        PathType path_type,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream);
private:
    //opencl stuff
    VerticalPathAggregation<-1, MAX_DISPARITY> m_down2up;
    VerticalPathAggregation<1, MAX_DISPARITY> m_up2down;
    HorizontalPathAggregation<-1, MAX_DISPARITY> m_right2left;
    HorizontalPathAggregation<1, MAX_DISPARITY> m_left2right;
    ObliquePathAggregation<1, 1, MAX_DISPARITY> m_upleft2downright;
    ObliquePathAggregation<-1, 1, MAX_DISPARITY> m_upright2downleft;
    ObliquePathAggregation<-1, -1, MAX_DISPARITY> m_downright2upleft;
    ObliquePathAggregation<1, -1, MAX_DISPARITY> m_downleft2upright;
    
};

}
}