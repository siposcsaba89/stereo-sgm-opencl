#include "sgm_details.h"
#include <regex>
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(ocl_sgm);



namespace sgm
{
namespace cl
{
SGMDetails::SGMDetails(cl_context ctx, cl_device_id device)
    : m_cl_context(ctx)
    , m_cl_device_id(device)
{
}
SGMDetails::~SGMDetails()
{
    if (m_kernel_check_consistency)
    {
        clReleaseKernel(m_kernel_check_consistency);
        m_kernel_check_consistency = nullptr;
    }
    if (m_kernel_median)
    {
        clReleaseKernel(m_kernel_median);
        m_kernel_median = nullptr;
    }
    if (m_kernel_disp_corr)
    {
        clReleaseKernel(m_kernel_disp_corr);
        clReleaseKernel(m_kernel_cast_16uto8u);
        m_kernel_disp_corr = nullptr;
        m_kernel_cast_16uto8u = nullptr;
    }
}

void SGMDetails::median_filter(const DeviceBuffer<uint16_t>& d_src,
    const DeviceBuffer<uint16_t>& d_dst,
    int width,
    int height,
    int pitch,
    cl_command_queue stream)
{
    if (nullptr == m_kernel_median)
    {
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_corr_disp_range = fs.open("src/ocl/median_filter.cl");
        auto kernel_src = std::string(kernel_corr_disp_range.begin(), kernel_corr_disp_range.end());
        m_program_median.init(m_cl_context, m_cl_device_id, kernel_src);

        m_kernel_median = m_program_median.getKernel("median3x3");
    }
    cl_int err = clSetKernelArg(m_kernel_median,
        0,
        sizeof(cl_mem),
        &d_src.data());
    err = clSetKernelArg(m_kernel_median, 1, sizeof(cl_mem), &d_dst.data());
    err = clSetKernelArg(m_kernel_median, 2, sizeof(width), &width);
    err = clSetKernelArg(m_kernel_median, 3, sizeof(height), &height);
    err = clSetKernelArg(m_kernel_median, 4, sizeof(pitch), &pitch);
    
    static constexpr int SIZE = 16;
    size_t local_size[2] = { SIZE, SIZE };
    size_t global_size[2] = {
        ((width + SIZE - 1) / SIZE) * local_size[0],
        ((height + SIZE - 1) / SIZE) * local_size[1]
    };

    err = clEnqueueNDRangeKernel(stream,
        m_kernel_median,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");
}

template<typename input_type>
inline void SGMDetails::check_consistency(DeviceBuffer<uint16_t>& d_left_disp,
    const DeviceBuffer<uint16_t>& d_right_disp,
    const DeviceBuffer<input_type>& d_src_left,
    int width, 
    int height,
    int src_pitch,
    int dst_pitch,
    bool subpixel,
    int LR_max_diff, 
    cl_command_queue stream)
{
    if (nullptr == m_kernel_check_consistency)
    {
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_check_consistency = fs.open("src/ocl/check_consistency.cl");
        auto kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_check_consistency.begin(), kernel_check_consistency.end());

        std::string kernel_template_types;
        if (std::is_same<input_type, uint16_t>::value)
        {
            kernel_template_types = "#define SRC_T uint16_t\n";
        }
        else if (std::is_same<input_type, uint8_t>::value)
        {
            kernel_template_types = "#define SRC_T uint8_t\n";
        }
        else
        {
            assert(false);
            throw std::runtime_error("Input image type must be 1 channel uint8_t or uint16_t");
        }

        std::string kernel_SUBPIXEL_SHIFT = "#define SUBPIXEL_SHIFT " + std::to_string(SubpixelShift()) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@SRC_T@"), kernel_template_types);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SUBPIXEL_SHIFT@"), kernel_SUBPIXEL_SHIFT);

        m_program_check_consistency.init(m_cl_context, m_cl_device_id, kernel_src);
        m_kernel_check_consistency = m_program_check_consistency.getKernel("check_consistency_kernel");
    }

    static constexpr int SIZE = 16;
    size_t local_size[2] = { SIZE, SIZE };
    size_t global_size[2] = {
        ((width + SIZE - 1) / SIZE) * local_size[0],
        ((height + SIZE - 1) / SIZE) * local_size[1]
    };
    cl_int err = clSetKernelArg(m_kernel_check_consistency,
        0,
        sizeof(cl_mem),
        &d_left_disp.data());
    err = clSetKernelArg(m_kernel_check_consistency, 1, sizeof(cl_mem), &d_right_disp.data());
    err = clSetKernelArg(m_kernel_check_consistency, 2, sizeof(cl_mem), &d_src_left.data());
    err = clSetKernelArg(m_kernel_check_consistency, 3, sizeof(width), &width);
    err = clSetKernelArg(m_kernel_check_consistency, 4, sizeof(height), &height);
    err = clSetKernelArg(m_kernel_check_consistency, 5, sizeof(src_pitch), &src_pitch);
    err = clSetKernelArg(m_kernel_check_consistency, 6, sizeof(dst_pitch), &dst_pitch);
    int sub_pixel_int = subpixel ? 1 : 0;
    err = clSetKernelArg(m_kernel_check_consistency, 7, sizeof(sub_pixel_int), &sub_pixel_int);
    err = clSetKernelArg(m_kernel_check_consistency, 8, sizeof(LR_max_diff), &LR_max_diff);


    err = clEnqueueNDRangeKernel(stream,
        m_kernel_check_consistency,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");
}

void SGMDetails::correct_disparity_range(DeviceBuffer<uint16_t>& d_disp,
    int width,
    int height,
    int pitch,
    bool subpixel,
    int min_disp,
    cl_command_queue stream)
{
    if (!subpixel && min_disp == 0)
    {
        return;
    }

    if (nullptr == m_kernel_disp_corr)
    {
        initDispRangeCorrection();
    };
    
    const int scale = subpixel ? SubpixelScale() : 1;
    const int min_disp_scaled = min_disp * scale;
    const int invalid_disp_scaled = (min_disp - 1) * scale;

    cl_int err = clSetKernelArg(m_kernel_disp_corr,
        0,
        sizeof(cl_mem),
        &d_disp.data());
    err = clSetKernelArg(m_kernel_disp_corr, 1, sizeof(width), &width);
    err = clSetKernelArg(m_kernel_disp_corr, 2, sizeof(height), &height);
    err = clSetKernelArg(m_kernel_disp_corr, 3, sizeof(pitch), &pitch);
    err = clSetKernelArg(m_kernel_disp_corr, 4, sizeof(min_disp_scaled), &min_disp_scaled);
    err = clSetKernelArg(m_kernel_disp_corr, 5, sizeof(invalid_disp_scaled), &invalid_disp_scaled);

    static constexpr int SIZE = 16;
    size_t local_size[2] = { SIZE, SIZE };
    size_t global_size[2] = {
        ((width + SIZE - 1) / SIZE) * local_size[0],
        ((height + SIZE - 1) / SIZE) * local_size[1]
    };

    err = clEnqueueNDRangeKernel(stream,
        m_kernel_disp_corr,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing correct disparity range kernel");
}

void SGMDetails::cast_16bit_8bit_array(const DeviceBuffer<uint16_t>& arr16bits,
    DeviceBuffer<uint8_t>& arr8bits, 
    int num_elements,
    cl_command_queue stream)
{
    if (nullptr == m_kernel_cast_16uto8u)
    {
        initDispRangeCorrection();
    }
    static constexpr int SIZE = 256;
    size_t local_size[1] = { SIZE };
    size_t global_size[1] = {
        ((num_elements + SIZE - 1) / SIZE) * local_size[0]
    };

    cl_int err = clEnqueueNDRangeKernel(stream,
        m_kernel_cast_16uto8u,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing winner_takes_all kernel");
}

void SGMDetails::initDispRangeCorrection()
{
    auto fs = cmrc::ocl_sgm::get_filesystem();
    auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
    auto kernel_corr_disp_range = fs.open("src/ocl/correct_disparity_range.cl");
    auto kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
        + std::string(kernel_corr_disp_range.begin(), kernel_corr_disp_range.end());
    m_program_disp_corr.init(m_cl_context, m_cl_device_id, kernel_src);

    m_kernel_disp_corr = m_program_disp_corr.getKernel("correct_disparity_range_kernel");
    m_kernel_cast_16uto8u = m_program_disp_corr.getKernel("cast_16bit_8bit_array_kernel");
}

// explicit instantiation of member function for types uint8_t, uint16_t
template void SGMDetails::check_consistency<uint8_t>(DeviceBuffer<uint16_t>& d_left_disp,
    const DeviceBuffer<uint16_t>& d_right_disp,
    const DeviceBuffer<uint8_t>& d_src_left,
    int width,
    int height,
    int src_pitch,
    int dst_pitch,
    bool subpixel,
    int LR_max_diff,
    cl_command_queue stream);

template void SGMDetails::check_consistency<uint16_t>(DeviceBuffer<uint16_t>& d_left_disp,
    const DeviceBuffer<uint16_t>& d_right_disp,
    const DeviceBuffer<uint16_t>& d_src_left,
    int width,
    int height,
    int src_pitch,
    int dst_pitch,
    bool subpixel,
    int LR_max_diff,
    cl_command_queue stream);

}
}
