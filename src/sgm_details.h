#pragma once

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

#include "libsgm_ocl/libsgm_ocl.h"
#include "libsgm_ocl/types.h"
#include "device_buffer.hpp"
#include "device_kernel.h"

namespace sgm
{
namespace cl
{

class SGMDetails
{
public:
    SGMDetails(cl_context ctx, cl_device_id device);
    ~SGMDetails();
    void median_filter(
        const DeviceBuffer<uint16_t>& d_src,
        const DeviceBuffer<uint16_t>& d_dst,
        int width,
        int height,
        int pitch,
        cl_command_queue stream);

    template <typename input_type>
    void check_consistency(
        DeviceBuffer<uint16_t>& d_left_disp,
        const DeviceBuffer<uint16_t>& d_right_disp,
        const DeviceBuffer<input_type>& d_src_left,
        int width,
        int height,
        int src_pitch,
        int dst_pitch,
        bool subpixel,
        int LR_max_diff,
        cl_command_queue stream);

    void correct_disparity_range(DeviceBuffer<uint16_t>& d_disp,
        int width,
        int height,
        int pitch,
        bool subpixel,
        int min_disp,
        cl_command_queue stream);

    void cast_16bit_8bit_array(const DeviceBuffer<uint16_t>& arr16bits,
        DeviceBuffer<uint8_t> & arr8bits,
        int num_elements,
        cl_command_queue stream);
private:
    void initDispRangeCorrection();
private:
    cl_context m_cl_context = nullptr;
    cl_device_id m_cl_device_id = nullptr;

    DeviceProgram m_program_median;
    cl_kernel m_kernel_median = nullptr;
    DeviceProgram m_program_check_consistency;
    cl_kernel m_kernel_check_consistency = nullptr;
    DeviceProgram m_program_disp_corr;
    cl_kernel m_kernel_disp_corr = nullptr;
    cl_kernel m_kernel_cast_16uto8u = nullptr;

private:
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int WARPS_PER_BLOCK = 8u;
    static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;
};

}

}
