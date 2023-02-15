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
template <typename input_type>
class SemiGlobalMatchingBase 
{
public:
    virtual void execute(DeviceBuffer<output_type>& dst_L,
        DeviceBuffer<output_type>& dst_R,
        const DeviceBuffer<input_type>& src_L,
        const DeviceBuffer<input_type>& src_R,
        DeviceBuffer<feature_type>& feature_buff_l,
        DeviceBuffer<feature_type>& feature_buff_r,
        cl_command_queue queue) = 0;

    virtual ~SemiGlobalMatchingBase() {}
};

template <typename input_type, int DISP_SIZE>
class SemiGlobalMatchingImpl : public SemiGlobalMatchingBase<input_type>
{
public:
    SemiGlobalMatchingImpl(cl_context ctx, cl_device_id device,
        int width,
        int height,
        int src_pitch,
        int dst_pitch,
        Parameters& param)
        : sgm_engine_(ctx, device, width, height, src_pitch, dst_pitch, param) {}
    void execute(
        DeviceBuffer<output_type> & dst_L,
        DeviceBuffer<output_type> & dst_R,
        const DeviceBuffer<input_type> & src_L,
        const DeviceBuffer<input_type>& src_R,
        DeviceBuffer<feature_type>& feature_buff_l,
        DeviceBuffer<feature_type>& feature_buff_r,
        cl_command_queue queue) override
    {
        sgm_engine_.enqueue(dst_L,
            dst_R,
            src_L,
            src_R,
            feature_buff_l,
            feature_buff_r,
            queue);
    }
    virtual ~SemiGlobalMatchingImpl() {}
private:
    SemiGlobalMatching<input_type, DISP_SIZE> sgm_engine_;
};

template <typename input_type>
struct CudaStereoSGMResources
{
    DeviceBuffer<feature_type> d_feature_buffer_left;
    DeviceBuffer<feature_type> d_feature_buffer_right;
    DeviceBuffer<uint8_t> d_src_left;
    DeviceBuffer<uint8_t> d_src_right;
    DeviceBuffer<uint16_t> d_left_disp;
    DeviceBuffer<uint16_t> d_right_disp;
    DeviceBuffer<uint16_t> d_tmp_left_disp;
    DeviceBuffer<uint16_t> d_tmp_right_disp;
    DeviceBuffer<uint8_t> d_u8_out_disp;

    std::unique_ptr<SemiGlobalMatchingBase<input_type>> sgm_engine;
    SGMDetails sgm_details;

    CudaStereoSGMResources(int width_,
        int height_,
        int disparity_size_,
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
        , sgm_details(ctx, device, input_type(0))
    {
        if (disparity_size_ == 64)
            sgm_engine = std::make_unique<SemiGlobalMatchingImpl<input_type, 64>>(ctx, device, width_, height_, src_pitch_, dst_pitch_, params);
        else if (disparity_size_ == 128)
            sgm_engine = std::make_unique<SemiGlobalMatchingImpl<input_type, 128>>(ctx, device, width_, height_, src_pitch_, dst_pitch_, params);
        else if (disparity_size_ == 256)
            sgm_engine = std::make_unique<SemiGlobalMatchingImpl<input_type, 256>>(ctx, device, width_, height_, src_pitch_, dst_pitch_, params);
        else
            throw std::logic_error("depth bits must be 8 or 16, and disparity size must be 64 or 128");

        d_feature_buffer_left .allocate(static_cast<size_t>(width_ * height_));
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
        sgm_engine.reset();
    }
};

template <typename input_type>
StereoSGM<input_type>::StereoSGM(int width,
    int height,
    int disparity_size,
    int output_depth_bits,
    cl_context ctx,
    cl_device_id cl_device,
    const Parameters& param) 
    : StereoSGM(width,
        height,
        disparity_size,
        output_depth_bits,
        width,
        width,
        ctx,
        cl_device,
        param)
{
}

static bool has_enough_depth(int output_depth_bits, int disparity_size, int min_disp, bool subpixel)
{
    // simulate minimum/maximum value
    int64_t max = static_cast<int64_t>(disparity_size) + min_disp - 1;
    if (subpixel)
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
        if (subpixel)
        {
            min *= SubpixelScale();
        }

        if (min < -(1ll << (output_depth_bits - 1))
            || 1ll << (output_depth_bits - 1) <= max)
            return false;
    }

    return true;
}

template <typename input_type>
StereoSGM<input_type>::StereoSGM(int width,
    int height,
    int disparity_size,
    int output_depth_bits,
    int src_pitch,
    int dst_pitch,
    cl_context ctx,
    cl_device_id cl_device,
    const Parameters& param)
    : m_width(width)
    , m_height(height)
    , m_max_disparity(disparity_size)
    , m_output_depth_bits(output_depth_bits)
    , m_src_pitch(src_pitch)
    , m_dst_pitch(dst_pitch)
    , m_cl_ctx(ctx)
    , m_cl_device(cl_device)
    , m_params(param)
{
    //create command queue
    initCL();
    // check values
    if (output_depth_bits != 8
        && output_depth_bits != 16) 
    {
        width = height = output_depth_bits = disparity_size = 0;
        throw std::logic_error("depth bits must be 8 or 16");
    }
    if (disparity_size != 64 && disparity_size != 128 && disparity_size != 256)
    {
        width = height = output_depth_bits = disparity_size = 0;
        throw std::logic_error("disparity size must be 64, 128 or 256");
    }
    if (!has_enough_depth(output_depth_bits, disparity_size, param.min_disp, param.subpixel))
    {
        width = height = output_depth_bits = disparity_size = 0;
        throw std::logic_error("output depth bits must be sufficient for representing output value");
    }
    if (param.path_type != PathType::SCAN_4PATH && param.path_type != PathType::SCAN_8PATH)
    {
        width = height = output_depth_bits = disparity_size = 0;
        throw std::logic_error("Path type must be PathType::SCAN_4PATH or PathType::SCAN_8PATH");
    }

    m_cu_res = std::make_unique<CudaStereoSGMResources<input_type>>(width,
        height,
        disparity_size,
        output_depth_bits,
        src_pitch,
        dst_pitch,
        m_params,
        m_cl_ctx,
        m_cl_device,
        m_cl_cmd_queue);

}
template <typename input_type>
StereoSGM<input_type>::~StereoSGM()
{
    m_cu_res.reset();
}

template <typename input_type>
void StereoSGM<input_type>::execute(const input_type* left_pixels,
    const input_type* right_pixels,
    uint16_t* dst)
{
    if (m_cu_res->d_src_left.size() == 0)
    {
        size_t size = m_src_pitch * m_height;
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
        0, //offset
        m_src_pitch * m_height * sizeof(input_type),
        left_pixels,
        0, nullptr, nullptr);

    err = clEnqueueWriteBuffer(m_cl_cmd_queue,
        m_cu_res->d_src_right.data(),
        false, // blocking
        0, //offset
        m_src_pitch * m_height * sizeof(input_type),
        right_pixels,
        0, nullptr, nullptr);
    execute(m_cu_res->d_src_left.data(), m_cu_res->d_src_right.data(), d_out_disp);

    err = clEnqueueReadBuffer(m_cl_cmd_queue,
        d_out_disp,
        true, // blocking
        0, //offset
        m_dst_pitch * m_height * m_output_depth_bits / 8,
        dst,
        0, nullptr, nullptr);
}

template<typename input_type>
void StereoSGM<input_type>::execute(cl_mem left_pixels, cl_mem right_pixels, cl_mem dst)
{
    DeviceBuffer<input_type> left_img(m_cl_ctx,
        m_src_pitch * m_height * sizeof(input_type),
        left_pixels);
    DeviceBuffer<input_type> right_img(m_cl_ctx,
        m_src_pitch * m_height * sizeof(input_type),
        right_pixels);
    DeviceBuffer<uint16_t> out_disp(m_cl_ctx,
        m_dst_pitch * m_height * sizeof(uint16_t),
        dst);
    DeviceBuffer<uint16_t>* left_disparity = &m_cu_res->d_left_disp;
    if (m_output_depth_bits == 16)
    {
        left_disparity = &out_disp;
    }

    m_cu_res->sgm_engine->execute(m_cu_res->d_tmp_left_disp,
        m_cu_res->d_tmp_right_disp,
        left_img,
        right_img,
        m_cu_res->d_feature_buffer_left,
        m_cu_res->d_feature_buffer_right,
        m_cl_cmd_queue);

    m_cu_res->sgm_details.median_filter(m_cu_res->d_tmp_left_disp,
        *left_disparity,
        m_width,
        m_height,
        m_dst_pitch,
        m_cl_cmd_queue
    );
    m_cu_res->sgm_details.median_filter(m_cu_res->d_tmp_right_disp,
        m_cu_res->d_right_disp,
        m_width,
        m_height,
        m_dst_pitch,
        m_cl_cmd_queue
    );
    m_cu_res->sgm_details.template check_consistency<input_type>(*left_disparity,
        m_cu_res->d_right_disp,
        left_img,
        m_width,
        m_height,
        m_src_pitch,
        m_dst_pitch,
        m_params.subpixel,
        m_params.LR_max_diff,
        m_cl_cmd_queue);
    m_cu_res->sgm_details.correct_disparity_range(*left_disparity,
        m_width,
        m_height,
        m_dst_pitch,
        m_params.subpixel,
        m_params.min_disp,
        m_cl_cmd_queue);
    if (m_output_depth_bits == 8)
    {
        DeviceBuffer<uint8_t> disparity(m_cl_ctx,
            m_dst_pitch * m_height * sizeof(uint8_t),
            dst);
        m_cu_res->sgm_details.cast_16bit_8bit_array(*left_disparity,
            disparity,
            m_dst_pitch * m_height,
            m_cl_cmd_queue);
    }
    clFinish(m_cl_cmd_queue);
}

template<typename input_type>
int StereoSGM<input_type>::get_invalid_disparity() const
{
    return (m_params.min_disp - 1) * (m_params.subpixel ? SubpixelScale() : 1);
}

template <typename input_type>
void StereoSGM<input_type>::initCL()
{
    cl_int err;
    m_cl_cmd_queue = clCreateCommandQueue(m_cl_ctx, m_cl_device, 0, &err);
    CHECK_OCL_ERROR(err, "Failed to create command queue");
}

template <typename input_type>
void StereoSGM<input_type>::finishQueue()
{
    cl_int err = clFinish(m_cl_cmd_queue);
    CHECK_OCL_ERROR(err, "Error finishing queue");
}

// explicit instantiate for uint8_t and uint16_t
template class StereoSGM<uint8_t>;
template class StereoSGM<uint16_t>;
}
}
