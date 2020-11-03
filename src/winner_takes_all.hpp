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

#include "libsgm_ocl/libsgm_ocl.h"
#include "libsgm_ocl/types.h"
#include "device_buffer.hpp"
#include "device_kernel.h"

namespace sgm
{
namespace cl
{

template <size_t MAX_DISPARITY>
class WinnerTakesAll 
{
public:
    WinnerTakesAll(cl_context ctx, cl_device_id device);

    const DeviceBuffer<uint16_t>& get_left_output() const
    {
        return m_left_buffer;
    }

    const DeviceBuffer<uint16_t>& get_right_output() const
    {
        return m_right_buffer;
    }

    void enqueue(
        const DeviceBuffer<uint8_t>& src,
        int width,
        int height,
        int pitch,
        float uniqueness,
        bool subpixel,
        PathType path_type,
        cl_command_queue stream);

    void enqueue(
        DeviceBuffer<uint16_t>& left,
        DeviceBuffer<uint16_t>& right,
        const DeviceBuffer<uint8_t>& src,
        int width,
        int height,
        int pitch,
        float uniqueness,
        bool subpixel,
        PathType path_type,
        cl_command_queue stream);
private:
    DeviceBuffer<uint16_t> m_left_buffer;
    DeviceBuffer<uint16_t> m_right_buffer;
    cl_context m_cl_context = nullptr;
    cl_device_id m_cl_device_id = nullptr;

    DeviceProgram m_program;
    cl_kernel m_kernel = nullptr;
private:
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int WARPS_PER_BLOCK = 8u;
    static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
};

}

}
