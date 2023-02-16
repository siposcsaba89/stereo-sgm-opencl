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
#include "device_buffer.hpp"
#include "libsgm_ocl/types.h"
#include "device_kernel.h"
#include <regex>
#include <vector>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ocl_sgm);

namespace sgm
{
namespace cl
{

struct VerticalPathAggregation
{
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;
    static constexpr unsigned int DP_BLOCK_SIZE = 16u;

    VerticalPathAggregation(
        cl_context ctx, cl_device_id device, int direction, MaxDisparity max_disparity);
    VerticalPathAggregation(const VerticalPathAggregation& o) = delete;
    VerticalPathAggregation(VerticalPathAggregation&& o) = default;
    VerticalPathAggregation& operator=(const VerticalPathAggregation& o) = delete;
    VerticalPathAggregation& operator=(VerticalPathAggregation&& o) = default;
    ~VerticalPathAggregation();

    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_kernel = nullptr;
    int m_direction;
    MaxDisparity m_max_disparity;
    void enqueue(DeviceBuffer<uint8_t>& dest,
        const DeviceBuffer<uint32_t>& left,
        const DeviceBuffer<uint32_t>& right,
        int width,
        int height,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream);

  private:
    void init();
};

struct HorizontalPathAggregation
{
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int DP_BLOCK_SIZE = 8u;
    static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;
    static constexpr unsigned int WARPS_PER_BLOCK = 4u;
    static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

    HorizontalPathAggregation(
        cl_context ctx, cl_device_id device, int direction, MaxDisparity max_disparity);
    HorizontalPathAggregation(const HorizontalPathAggregation& o) = delete;
    HorizontalPathAggregation(HorizontalPathAggregation&& o) = default;
    HorizontalPathAggregation& operator=(const HorizontalPathAggregation& o) = delete;
    HorizontalPathAggregation& operator=(HorizontalPathAggregation&& o) = default;
    ~HorizontalPathAggregation();
    void enqueue(DeviceBuffer<uint8_t>& dest,
        const DeviceBuffer<uint32_t>& left,
        const DeviceBuffer<uint32_t>& right,
        int width,
        int height,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream);

  private:
    void init();

  private:
    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_kernel = nullptr;
    int m_direction;
    MaxDisparity m_max_disparity;
};

struct ObliquePathAggregation
{
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int DP_BLOCK_SIZE = 16u;
    static constexpr unsigned int BLOCK_SIZE = WARP_SIZE * 8u;

    ObliquePathAggregation(cl_context ctx,
        cl_device_id device,
        int x_direction,
        int y_direction,
        MaxDisparity max_disparity);
    ObliquePathAggregation(const ObliquePathAggregation& o) = delete;
    ObliquePathAggregation(ObliquePathAggregation&& o) = default;
    ObliquePathAggregation& operator=(const ObliquePathAggregation& o) = delete;
    ObliquePathAggregation& operator=(ObliquePathAggregation&& o) = default;
    ~ObliquePathAggregation();
    void enqueue(DeviceBuffer<uint8_t>& dest,
        const DeviceBuffer<uint32_t>& left,
        const DeviceBuffer<uint32_t>& right,
        int width,
        int height,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream);

  private:
    void init();

  private:
    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_kernel = nullptr;
    int m_x_direction;
    int m_y_direction;
    MaxDisparity m_max_disparity;
};

class PathAggregation
{
  private:
    static const unsigned int MAX_NUM_PATHS = 8;

    DeviceBuffer<uint8_t> m_cost_buffer;
    std::vector<DeviceBuffer<uint8_t>> m_sub_buffers;
    cl_command_queue m_streams[MAX_NUM_PATHS];

  public:
    PathAggregation(cl_context ctx,
        cl_device_id device,
        PathType path_type,
        MaxDisparity max_disparity,
        int width,
        int height);
    ~PathAggregation();

    const DeviceBuffer<uint8_t>& get_output() const;

    void enqueue(const DeviceBuffer<uint32_t>& left,
        const DeviceBuffer<uint32_t>& right,
        unsigned int p1,
        unsigned int p2,
        int min_disp,
        cl_command_queue stream);

  private:
    // opencl stuff
    VerticalPathAggregation m_down2up;
    VerticalPathAggregation m_up2down;
    HorizontalPathAggregation m_right2left;
    HorizontalPathAggregation m_left2right;
    ObliquePathAggregation m_upleft2downright;
    ObliquePathAggregation m_upright2downleft;
    ObliquePathAggregation m_downright2upleft;
    ObliquePathAggregation m_downleft2upright;
    PathType m_path_type;
    MaxDisparity m_max_disparity;
    int m_width;
    int m_height;
};

} // namespace cl
} // namespace sgm