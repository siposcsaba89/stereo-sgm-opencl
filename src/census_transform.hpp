#pragma once
#include "libsgm_ocl/types.h"
#include "device_buffer.hpp"
#include "device_kernel.h"
#include <regex>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ocl_sgm);

namespace sgm
{
namespace cl
{
class CensusTransform
{
    static constexpr unsigned int BLOCK_SIZE = 128;
    static constexpr unsigned int WINDOW_WIDTH = 9;
    static constexpr unsigned int WINDOW_HEIGHT = 7;
    static constexpr unsigned int LINES_PER_BLOCK = 16;


public:
    CensusTransform(cl_context ctx,
        cl_device_id device,
        uint32_t input_bits);
    ~CensusTransform();

    void enqueue(
        const cl_mem src,
        DeviceBuffer<uint32_t>& feature_buffer,
        int width,
        int height,
        int pitch,
        cl_command_queue stream);

private:
    DeviceProgram m_program;
    cl_context m_cl_ctx = nullptr;
    cl_device_id m_cl_device = nullptr;
    cl_kernel m_census_kernel = nullptr;
};


CensusTransform::CensusTransform(cl_context ctx,
    cl_device_id device, uint32_t input_bits)
    : m_cl_ctx(ctx)
    , m_cl_device(device)
{
    std::string kernel_template_types;

    if (input_bits == 16)
    {
        kernel_template_types = "#define pixel_type uint16_t\n";
    }
    else if (input_bits == 8)
    {
        kernel_template_types = "#define pixel_type uint8_t\n";
    }
    else
    {
        assert(false);
        throw std::runtime_error("Input image type must be 1 channel uint8_t or uint16_t");
    }
    //resource reading
    auto fs = cmrc::ocl_sgm::get_filesystem();
    auto kernel_rc = fs.open("src/ocl/census.cl");
    auto kernel = std::string(kernel_rc.begin(), kernel_rc.end());
    std::regex px_type_regex("@pixel_type@");
    kernel = std::regex_replace(kernel, px_type_regex, kernel_template_types);
    m_program.init(m_cl_ctx, m_cl_device, kernel);

    m_census_kernel = m_program.getKernel("census_transform_kernel");
    
}

CensusTransform::~CensusTransform()
{
    if (m_census_kernel)
    {
        clReleaseKernel(m_census_kernel);
        m_census_kernel = nullptr;
    }
}

void CensusTransform::enqueue(const cl_mem src,
    DeviceBuffer<uint32_t> & feature_buffer,
    int width,
    int height,
    int pitch,
    cl_command_queue stream)
{
    cl_int err = clSetKernelArg(m_census_kernel,
        0,
        sizeof(cl_mem),
        &feature_buffer.data());
    err = clSetKernelArg(m_census_kernel, 1, sizeof(cl_mem), &src);
    err = clSetKernelArg(m_census_kernel, 2, sizeof(width), &width);
    err = clSetKernelArg(m_census_kernel, 3, sizeof(height), &height);
    err = clSetKernelArg(m_census_kernel, 4, sizeof(pitch), &pitch);

    const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
    const int height_per_block = LINES_PER_BLOCK;

    //setup kernels
    size_t global_size[2] = {
        (size_t)((width + width_per_block - 1) / width_per_block * BLOCK_SIZE),
        (size_t)((height + height_per_block - 1) / height_per_block)
    };
    size_t local_size[2] = { BLOCK_SIZE, 1 };
    err = clEnqueueNDRangeKernel(stream,
        m_census_kernel,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing census kernel");
}

}
}
