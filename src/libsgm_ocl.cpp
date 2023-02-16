#include "libsgm_ocl/libsgm_ocl.h"
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include "sgm.hpp"
#include "device_buffer.hpp"
#include "sgm_details.h"
#include <memory>

namespace sgm
{
namespace cl
{
struct CudaStereoSGMResources
{
    DeviceBuffer<uint32_t> d_feature_buffer_left;
    DeviceBuffer<uint32_t> d_feature_buffer_right;
    DeviceBuffer<uint8_t> d_src_left;
    DeviceBuffer<uint8_t> d_src_right;
    DeviceBuffer<uint16_t> d_left_disp;
    DeviceBuffer<uint16_t> d_right_disp;
    DeviceBuffer<uint16_t> d_tmp_left_disp;
    DeviceBuffer<uint16_t> d_tmp_right_disp;
    DeviceBuffer<uint8_t> d_u8_out_disp;

    SemiGlobalMatching sgm_engine;
    SGMDetails sgm_details;

    CudaStereoSGMResources(int width_,
        int height_,
        MaxDisparity max_disparity,
        int input_bits,
        int output_depth_bits_,
        int src_pitch_,
        int dst_pitch_,
        Parameters params,
        cl_context ctx,
        cl_device_id device,
        cl_command_queue queue)
        : d_feature_buffer_left(ctx)
        , d_feature_buffer_right(ctx)
        , d_src_left(ctx)
        , d_src_right(ctx)
        , d_left_disp(ctx)
        , d_right_disp(ctx)
        , d_tmp_left_disp(ctx)
        , d_tmp_right_disp(ctx)
        , d_u8_out_disp(ctx)
        , sgm_engine(ctx,
              device,
              max_disparity,
              width_,
              height_,
              src_pitch_,
              dst_pitch_,
              input_bits,
              params)
        , sgm_details(ctx, device, input_bits)
    {

        d_feature_buffer_left.allocate(static_cast<size_t>(width_ * height_));
        d_feature_buffer_right.allocate(static_cast<size_t>(width_ * height_));

        this->d_left_disp.allocate(dst_pitch_ * height_);
        this->d_right_disp.allocate(dst_pitch_ * height_);

        this->d_tmp_left_disp.allocate(dst_pitch_ * height_);
        this->d_tmp_right_disp.allocate(dst_pitch_ * height_);

        this->d_left_disp.fillZero(queue);
        this->d_right_disp.fillZero(queue);
        this->d_tmp_left_disp.fillZero(queue);
        this->d_tmp_right_disp.fillZero(queue);
    }

    ~CudaStereoSGMResources()
    {
    }
};

StereoSGM::StereoSGM(int width,
    int height,
    MaxDisparity disparity_size,
    int input_bits,
    int output_depth_bits,
    cl_context ctx,
    cl_device_id cl_device,
    const Parameters& param)
    : StereoSGM(width,
          height,
          disparity_size,
          input_bits,
          output_depth_bits,
          width,
          width,
          ctx,
          cl_device,
          param)
{
}

static bool has_enough_depth(
    int output_depth_bits, MaxDisparity disparity_size, int min_disp, DispPrecision subpixel)
{
    // simulate minimum/maximum value
    int64_t max = static_cast<int64_t>(disparity_size) + min_disp - 1;
    if (subpixel == DispPrecision::SUBPIXEL)
    {
        max *= SubpixelScale();
        max += SubpixelScale() - 1;
    }

    if (1ll << output_depth_bits <= max)
        return false;

    if (min_disp <= 0)
    {
        // whether or not output can be represented by signed
        int64_t min = static_cast<int64_t>(min_disp) - 1;
        if (subpixel == DispPrecision::SUBPIXEL)
        {
            min *= SubpixelScale();
        }

        if (min < -(1ll << (output_depth_bits - 1)) || 1ll << (output_depth_bits - 1) <= max)
            return false;
    }

    return true;
}

StereoSGM::StereoSGM(int width,
    int height,
    MaxDisparity disparity_size,
    int input_bits,
    int output_depth_bits,
    int src_pitch,
    int dst_pitch,
    cl_context ctx,
    cl_device_id cl_device,
    const Parameters& param)
    : m_width(width)
    , m_height(height)
    , m_max_disparity(disparity_size)
    , m_input_bits(input_bits)
    , m_output_depth_bits(output_depth_bits)
    , m_src_pitch(src_pitch)
    , m_dst_pitch(dst_pitch)
    , m_cl_ctx(ctx)
    , m_cl_device(cl_device)
    , m_params(param)
{
    // create command queue
    initCL();
    // check values
    if (output_depth_bits != 8 && output_depth_bits != 16)
    {
        throw std::logic_error("depth bits must be 8 or 16");
    }
    if (!has_enough_depth(output_depth_bits, disparity_size, param.min_disp, param.subpixel))
    {
        throw std::logic_error(
            "output depth bits must be sufficient for representing output value");
    }
    if (param.path_type != PathType::SCAN_4PATH && param.path_type != PathType::SCAN_8PATH)
    {
        throw std::logic_error("Path type must be PathType::SCAN_4PATH or PathType::SCAN_8PATH");
    }

    m_cu_res = std::make_unique<CudaStereoSGMResources>(width,
        height,
        disparity_size,
        input_bits,
        output_depth_bits,
        src_pitch,
        dst_pitch,
        m_params,
        m_cl_ctx,
        m_cl_device,
        m_cl_cmd_queue);
}
StereoSGM::~StereoSGM()
{
    m_cu_res.reset();
}

void StereoSGM::execute(const void* left_pixels, const void* right_pixels, uint16_t* dst)
{
    int input_bytes = m_input_bits / 8;
    if (m_cu_res->d_src_left.size() == 0)
    {
        size_t size = m_src_pitch * m_height * input_bytes;
        m_cu_res->d_src_left.allocate(size);
        m_cu_res->d_src_right.allocate(size);
    }
    cl_mem d_out_disp = m_cu_res->d_left_disp.data();
    if (m_output_depth_bits == 8 && m_cu_res->d_u8_out_disp.size() == 0)
    {
        m_cu_res->d_u8_out_disp.allocate(m_dst_pitch * m_height);
        d_out_disp = m_cu_res->d_u8_out_disp.data();
    }

    cl_int err = clEnqueueWriteBuffer(m_cl_cmd_queue,
        m_cu_res->d_src_left.data(),
        false, // blocking
        0,     // offset
        m_src_pitch * m_height * input_bytes,
        left_pixels,
        0,
        nullptr,
        nullptr);

    err = clEnqueueWriteBuffer(m_cl_cmd_queue,
        m_cu_res->d_src_right.data(),
        false, // blocking
        0,     // offset
        m_src_pitch * m_height * input_bytes,
        right_pixels,
        0,
        nullptr,
        nullptr);
    execute(m_cu_res->d_src_left.data(), m_cu_res->d_src_right.data(), d_out_disp);

    err = clEnqueueReadBuffer(m_cl_cmd_queue,
        d_out_disp,
        true, // blocking
        0,    // offset
        m_dst_pitch * m_height * m_output_depth_bits / 8,
        dst,
        0,
        nullptr,
        nullptr);
}

void StereoSGM::execute(cl_mem left_pixels, cl_mem right_pixels, cl_mem dst)
{
    DeviceBuffer<uint16_t> out_disp(m_cl_ctx, m_dst_pitch * m_height * sizeof(uint16_t), dst);
    DeviceBuffer<uint16_t>* left_disparity = &m_cu_res->d_left_disp;
    if (m_output_depth_bits == 16)
    {
        left_disparity = &out_disp;
    }

    m_cu_res->sgm_engine.enqueue(m_cu_res->d_tmp_left_disp,
        m_cu_res->d_tmp_right_disp,
        left_pixels,
        right_pixels,
        m_cu_res->d_feature_buffer_left,
        m_cu_res->d_feature_buffer_right,
        m_cl_cmd_queue);

    m_cu_res->sgm_details.median_filter(
        m_cu_res->d_tmp_left_disp, *left_disparity, m_width, m_height, m_dst_pitch, m_cl_cmd_queue);
    m_cu_res->sgm_details.median_filter(m_cu_res->d_tmp_right_disp,
        m_cu_res->d_right_disp,
        m_width,
        m_height,
        m_dst_pitch,
        m_cl_cmd_queue);
    m_cu_res->sgm_details.check_consistency(*left_disparity,
        m_cu_res->d_right_disp,
        left_pixels,
        m_width,
        m_height,
        m_src_pitch,
        m_dst_pitch,
        m_params.subpixel == DispPrecision::SUBPIXEL,
        m_params.LR_max_diff,
        m_cl_cmd_queue);
    m_cu_res->sgm_details.correct_disparity_range(*left_disparity,
        m_width,
        m_height,
        m_dst_pitch,
        m_params.subpixel == DispPrecision::SUBPIXEL,
        m_params.min_disp,
        m_cl_cmd_queue);
    if (m_output_depth_bits == 8)
    {
        DeviceBuffer<uint8_t> disparity(m_cl_ctx, m_dst_pitch * m_height * sizeof(uint8_t), dst);
        m_cu_res->sgm_details.cast_16bit_8bit_array(
            *left_disparity, disparity, m_dst_pitch * m_height, m_cl_cmd_queue);
    }
    clFinish(m_cl_cmd_queue);
}

int StereoSGM::get_invalid_disparity() const
{
    return (m_params.min_disp - 1) *
           (m_params.subpixel == DispPrecision::SUBPIXEL ? SubpixelScale() : 1);
}

void StereoSGM::initCL()
{
    cl_int err;
    m_cl_cmd_queue = clCreateCommandQueue(m_cl_ctx, m_cl_device, 0, &err);
    CHECK_OCL_ERROR(err, "Failed to create command queue");
}

void StereoSGM::finishQueue()
{
    cl_int err = clFinish(m_cl_cmd_queue);
    CHECK_OCL_ERROR(err, "Error finishing queue");
}

} // namespace cl
} // namespace sgm
