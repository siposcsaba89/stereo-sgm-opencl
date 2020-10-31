#pragma once
#include "types.hpp"
#include "device_buffer.hpp"
#include "device_kernel.h"
#include <regex>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ocl_sgm);


#define WINDOW_WIDTH  9
#define WINDOW_HEIGHT  7
#define BLOCK_SIZE 128
#define LINES_PER_BLOCK 16

namespace sgm
{
namespace cl
{

template <typename input_type>
class CensusTransform
{
public:
    CensusTransform(cl_context ctx,
        cl_device_id device);
    ~CensusTransform();
    const cl_mem get_output() const {
        return m_feature_buffer.data();
    }

    void enqueue(
        const DeviceBuffer<input_type> & src,
        DeviceBuffer<feature_type>& feature_buffer,
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


template<typename input_type>
inline CensusTransform<input_type>::CensusTransform(cl_context ctx,
    cl_device_id device)
    : m_cl_ctx(ctx)
    , m_cl_device(device)
{
}

template<typename input_type>
inline CensusTransform<input_type>::~CensusTransform()
{
    clReleaseKernel(m_census_kernel);
}

template<typename input_type>
inline void CensusTransform<input_type>::enqueue(const DeviceBuffer<input_type> & src,
    DeviceBuffer<feature_type> & feature_buffer,
    int width,
    int height,
    int pitch,
    cl_command_queue stream)
{
    if (m_census_kernel == nullptr)
    {
        std::string kernel_template_types;

        if (std::is_same<input_type, uint16_t>::value)
        {
            kernel_template_types = "#define pixel_type uint8_t\n";
        }
        else if (std::is_same<input_type, uint8_t>::value)
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
        //DEBUG
        std::cout << "libsgm_ocl / census.cl" << std::endl;
        std::cout << kernel << std::endl;

        m_census_kernel = m_program.getKernel("census_transform_kernel");
    }

    cl_int err = clSetKernelArg(m_census_kernel,
        0,
        sizeof(cl_mem),
        &feature_buffer.data());
    err = clSetKernelArg(m_census_kernel, 1, sizeof(cl_mem), &src.data());
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