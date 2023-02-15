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
#include <memory>
#include <cstdint>
#include "libsgm_ocl/libsgm_ocl.h"
#include "device_buffer.hpp"

namespace sgm
{
namespace cl
{

template <typename input_type, size_t MAX_DISPARITY>
class SemiGlobalMatching
{
public:
    using output_type = sgm::cl::output_type;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;

public:
    SemiGlobalMatching(cl_context context,
        cl_device_id device,
        int width,
        int height,
        int src_pitch,
        int dst_pitch,
        const Parameters& param);
    ~SemiGlobalMatching();

    void enqueue(
        DeviceBuffer<output_type>& dest_left,
        DeviceBuffer<output_type>& dest_right,
        const DeviceBuffer<input_type> & src_left,
        const DeviceBuffer<input_type> & src_right,
        DeviceBuffer<feature_type>& feature_buffer_left,
        DeviceBuffer<feature_type>& feature_buffer_right,
        cl_command_queue stream);
};

}
}
