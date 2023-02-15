#pragma once

#include <CL/cl.h>
#include <string>
#include <iostream>

namespace sgm
{
namespace cl
{
class DeviceProgram
{
public:
    DeviceProgram() = default;
    DeviceProgram(cl_context ctx,
        cl_device_id device,
        const std::string& kernel_str)
    {
        init(ctx, device, kernel_str);
    }

    void init(cl_context ctx,
        cl_device_id device,
        const std::string& kernel_str)
    {
        if (m_cl_program != nullptr)
        {
            clReleaseProgram(m_cl_program);
            m_cl_program = nullptr;
        }

        cl_int err;
        const char* kernel_src = kernel_str.c_str();
        size_t kenel_src_length = kernel_str.size();
        m_cl_program = clCreateProgramWithSource(ctx, 1, &kernel_src, &kenel_src_length, &err);
        err = clCompileProgram(m_cl_program, 1, &device, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
        m_cl_program = clLinkProgram(ctx, 1, &device, nullptr, 1, &m_cl_program, nullptr, nullptr, &err);
        //err = clBuildProgram(m_cl_program, 1, &device, nullptr, nullptr, nullptr);

        size_t build_log_size = 0;
        clGetProgramBuildInfo(m_cl_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size);
        std::string build_log;
        build_log.resize(build_log_size);
        clGetProgramBuildInfo(m_cl_program, device, CL_PROGRAM_BUILD_LOG,
            build_log_size,
            &build_log[0],
            nullptr);
        if (build_log.size() > 10)
            std::cout << "OpenCL build info: " << build_log << std::endl;
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Cannot build ocl program!");
        }
    }

    cl_kernel getKernel(const std::string & name)
    {
        cl_int err;
        cl_kernel ret = clCreateKernel(m_cl_program, name.c_str(), &err);
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Cannot find ocl kernel: " + name);
        }
        return ret;
    }

    ~DeviceProgram()
    {
        clReleaseProgram(m_cl_program);
    }

    bool isInitialized() const
    {
        return m_cl_program != nullptr;
    }

private:
    cl_program m_cl_program = nullptr;
};
}
}
