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

#include <cstddef>
#include <CL/cl.h>
#include <iostream>

#define CHECK_OCL_ERROR(err, msg) \
    if (err != CL_SUCCESS) \
    { \
        std::cout << "OCL_ERROR at line " << __LINE__ << ". Message: " << msg << std::endl; \
    }\


namespace sgm 
{
namespace cl
{

template <typename value_type>
class DeviceBuffer
{
public:
    DeviceBuffer(cl_context ctx = nullptr)
        : m_cl_ctx(ctx)
        , m_data(nullptr)
        , m_size(0)
        , m_owns_data(false)
    { }

    explicit DeviceBuffer(cl_context ctx, size_t n)
        : m_cl_ctx(ctx)
        , m_data(nullptr)
        , m_size(0)
        , m_owns_data(false)
    {
        allocate(n);
    }

    explicit DeviceBuffer(cl_context ctx, size_t n, cl_mem data)
        : m_cl_ctx(ctx)
        , m_data(data)
        , m_size(n)
        , m_owns_data(false)
    {
    }

    void setBufferData(cl_context ctx, size_t n, cl_mem data)
    {
        m_cl_ctx = ctx;
        m_size = n;
        m_data = data;
        m_owns_data = false;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& obj)
        : m_data(obj.m_data)
        , m_size(obj.m_size)
    {
        obj.m_data = nullptr;
        obj.m_size = 0;
    }

    ~DeviceBuffer() {
        destroy();
    }


    void allocate(size_t n) {
        if (m_data && m_size >= n)
            return;

        destroy();
        cl_int err;
        m_data = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(value_type) * n, nullptr, &err);
        CHECK_OCL_ERROR(err, "Allocating device buffer");
        m_size = n;
        m_owns_data = true;
    }

    void destroy()
    {
        if (m_data && m_owns_data)
        {
            cl_int err = clReleaseMemObject(m_data);
            CHECK_OCL_ERROR(err, "Destroying device buffer");
            m_data = nullptr;
        }

        m_data = nullptr;
        m_size = 0;
    }

    void fillZero(cl_command_queue queue)
    {
        value_type pattern = 0;
        clEnqueueFillBuffer(queue,
            m_data,
            &pattern,
            sizeof(pattern),
            0,
            m_size * sizeof(value_type),
            0,
            nullptr,
            nullptr);
    }

    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(DeviceBuffer&& obj) {
        m_data = obj.m_data;
        m_size = obj.m_size;
        obj.m_data = nullptr;
        obj.m_size = 0;
        return *this;
    }

    size_t size() const {
        return m_size;
    }

    cl_mem & data()
    {
        return m_data;
    }

    const cl_mem& data() const
    {
        return m_data;
    }
private:
    cl_context m_cl_ctx = nullptr;
    cl_mem m_data = nullptr;
    size_t m_size = 0;
    bool m_owns_data = false;
};

}
}

