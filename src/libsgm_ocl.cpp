#include "libsgm_ocl/libsgm_ocl.h"
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include "sgm.hpp"
#include "device_buffer.hpp"
//#include <cmrc/cmrc.hpp>
//CMRC_DECLARE(ocl_sgm);

//resource reading
//auto       fs = cmrc::ocl_sgm::get_filesystem();
//auto       flower_rc = fs.open("libsgm_ocl/sgm.cl");
//const auto rc_size = std::distance(flower_rc.begin(), flower_rc.end());
//auto kernel = std::string(flower_rc.begin(), flower_rc.end());
//std::cout << kernel << std::endl;
//std::cout << rc_size << std::endl;

namespace sgm
{
namespace cl
{
static const int SUBPIXEL_SHIFT = 4;
static const int SUBPIXEL_SCALE = (1 << SUBPIXEL_SHIFT);

template <typename input_type>
class SemiGlobalMatchingBase 
{
public:
    using output_type = output_type;
    virtual void execute(DeviceBuffer<output_type>& dst_L,
        DeviceBuffer<output_type>& dst_R,
        const DeviceBuffer<input_type>& src_L,
        const DeviceBuffer<input_type>& src_R,
        DeviceBuffer<feature_type>& feature_buff_l,
        DeviceBuffer<feature_type>& feature_buff_r,
        int w,
        int h,
        int sp,
        int dp,
        Parameters& param,
        cl_command_queue queue) = 0;

    virtual ~SemiGlobalMatchingBase() {}
};

template <typename input_type, int DISP_SIZE>
class SemiGlobalMatchingImpl : public SemiGlobalMatchingBase<input_type>
{
public:
    SemiGlobalMatchingImpl(cl_context ctx, cl_device_id device)
        : sgm_engine_(ctx, device) {}
    void execute(
        DeviceBuffer<output_type> & dst_L,
        DeviceBuffer<output_type> & dst_R,
        const DeviceBuffer<input_type> & src_L,
        const DeviceBuffer<input_type>& src_R,
        DeviceBuffer<feature_type>& feature_buff_l,
        DeviceBuffer<feature_type>& feature_buff_r,
        int w,
        int h,
        int sp, 
        int dp, 
        Parameters& param,
        cl_command_queue queue) override
    {
        sgm_engine_.enqueue(dst_L,
            dst_R,
            src_L,
            src_R,
            feature_buff_l,
            feature_buff_r,
            w, 
            h,
            sp,
            dp,
            param,
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



    std::unique_ptr<SemiGlobalMatchingBase<input_type>> sgm_engine;

    CudaStereoSGMResources(int width_,
        int height_,
        int disparity_size_,
        int output_depth_bits_,
        int src_pitch_,
        int dst_pitch_,
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
    {

        if (disparity_size_ == 64)
            sgm_engine = std::make_unique<SemiGlobalMatchingImpl<input_type, 64>>(ctx, device);
        else if (disparity_size_ == 128)
            sgm_engine = std::make_unique<SemiGlobalMatchingImpl<input_type, 128>>(ctx, device);
        else if (disparity_size_ == 256)
            sgm_engine = std::make_unique<SemiGlobalMatchingImpl<input_type, 256>>(ctx, device);
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
        max *= sgm::cl::SUBPIXEL_SCALE;
        max += sgm::cl::SUBPIXEL_SCALE - 1;
    }

    if (1ll << output_depth_bits <= max)
        return false;

    if (min_disp <= 0)
    {
        // whether or not output can be represented by signed
        int64_t min = static_cast<int64_t>(min_disp) - 1;
        if (subpixel)
        {
            min *= sgm::cl::SUBPIXEL_SCALE;
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

    m_cu_res = new CudaStereoSGMResources<input_type>(width,
        height,
        disparity_size,
        output_depth_bits,
        src_pitch,
        dst_pitch,
        m_cl_ctx,
        m_cl_device,
        m_cl_cmd_queue);

}
template <typename input_type>
StereoSGM<input_type>::~StereoSGM()
{
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
    execute(m_cu_res->d_src_left.data(), m_cu_res->d_src_right.data(), m_cu_res->d_left_disp.data());

    err = clEnqueueReadBuffer(m_cl_cmd_queue,
        m_cu_res->d_tmp_left_disp.data(),
        true, // blocking
        0, //offset
        m_dst_pitch * m_height * sizeof(uint16_t),
        dst,
        0, nullptr, nullptr);
    finishQueue();

    //todo support uint8_t output
    //if (!is_cuda_output(inout_type_) && output_depth_bits_ == 8) {
    //    sgm::details::cast_16bit_8bit_array((const uint16_t*)d_left_disp, (uint8_t*)d_tmp_left_disp, dst_pitch_ * height_);
    //    CudaSafeCall(cudaMemcpy(dst, d_tmp_left_disp, sizeof(uint8_t) * dst_pitch_ * height_, cudaMemcpyDeviceToHost));
    //}
    //else if (is_cuda_output(inout_type_) && output_depth_bits_ == 8) {
    //    sgm::details::cast_16bit_8bit_array((const uint16_t*)d_left_disp, (uint8_t*)dst, dst_pitch_ * height_);
    //}

    //    cl_int err;
    //
//    //d_src_left->write(left_data);
//    err = clEnqueueWriteBuffer(m_cl_cmd_queue,
//        d_src_left,
//        false, // blocking
//        0, //offset
//        m_width * m_height,
//        left_data,
//        0, nullptr, nullptr);
//
//    //d_src_right->write(right_data);
//    err = clEnqueueWriteBuffer(m_cl_cmd_queue,
//        d_src_right,
//        false, // blocking
//        0, //offset
//        m_width * m_height,
//        right_data,
//        0, nullptr, nullptr);
//
//    census();
//
//    mem_init();
//    //m_context->finish(0);
//    //m_context->finish(0);
//
//
//    //(*m_copy_u8_to_u16)(0,
//    //	m_width * m_height,
//    //	128);
//    //m_context->finish(0);
//
//    path_aggregation();
//    finishQueue();
//
//
//    winner_takes_all();
//    //m_context->finish(0);
//
//    median();
//    //m_context->finish(0);
//
//    check_consistency_left();
//    finishQueue();
//

//
//    //d_tmp_left_disp->read(output_buffer);
//    //m_context->finish(0);
//
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
    DeviceBuffer<input_type> disparity(m_cl_ctx,
        m_dst_pitch * m_height * sizeof(uint16_t),
        dst);

    m_cu_res->sgm_engine->execute(m_cu_res->d_tmp_left_disp,
        m_cu_res->d_tmp_right_disp,
        left_img,
        right_img,
        m_cu_res->d_feature_buffer_left,
        m_cu_res->d_feature_buffer_right,
        m_width,
        m_height,
        m_src_pitch,
        m_dst_pitch,
        m_params,
        m_cl_cmd_queue);

}

template<typename input_type>
int StereoSGM<input_type>::get_invalid_disparity() const
{
    return (m_params.min_disp - 1) * (m_params.subpixel ? SUBPIXEL_SCALE : 1);
}

template <typename input_type>
void StereoSGM<input_type>::initCL()
{
    cl_int err;
    m_cl_cmd_queue = clCreateCommandQueue(m_cl_ctx, m_cl_device, 0, &err);
    CHECK_OCL_ERROR(err, "Failed to create command queue");


    //path aggregation kernels
    //m_aggregate_vertical_path_kernel_dir_1 = clCreateKernel(sgm_program, "aggregate_vertical_path_kernel", &err);
    //CHECK_OCL_ERROR(err, "Create aggregate_vertical_path_kernel");
    //m_aggregate_vertical_path_kernel_dir__1 = clCreateKernel(sgm_program, "aggregate_vertical_path_kernel_down2up", &err);
    //CHECK_OCL_ERROR(err, "Create aggregate_vertical_path_kernel_down2up");
    //m_compute_stereo_horizontal_dir_kernel_4 = clCreateKernel(sgm_program, "compute_stereo_horizontal_dir_kernel_4", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_horizontal_dir_kernel_4");
    
    //m_compute_stereo_vertical_dir_kernel_2 =  clCreateKernel(sgm_program, "compute_stereo_vertical_dir_kernel_2", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_vertical_dir_kernel_2");
    //m_compute_stereo_vertical_dir_kernel_6 =  clCreateKernel(sgm_program, "compute_stereo_vertical_dir_kernel_6", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_vertical_dir_kernel_6");
    //
    //m_compute_stereo_oblique_dir_kernel_1 =  clCreateKernel(sgm_program, "compute_stereo_oblique_dir_kernel_1", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_oblique_dir_kernel_1");
    //m_compute_stereo_oblique_dir_kernel_3 =  clCreateKernel(sgm_program, "compute_stereo_oblique_dir_kernel_3", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_oblique_dir_kernel_3");
    //m_compute_stereo_oblique_dir_kernel_5 =  clCreateKernel(sgm_program, "compute_stereo_oblique_dir_kernel_5", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_oblique_dir_kernel_5");
    //m_compute_stereo_oblique_dir_kernel_7 =  clCreateKernel(sgm_program, "compute_stereo_oblique_dir_kernel_7", &err);
    //CHECK_OCL_ERROR(err, "Create compute_stereo_oblique_dir_kernel_7");
    //
    //
    //m_winner_takes_all_kernel128 = clCreateKernel(sgm_program, "winner_takes_all_kernel128", &err);
    //CHECK_OCL_ERROR(err, "Create winner_takes_all_kernel128");
    //
    //m_check_consistency_left = clCreateKernel(sgm_program, "check_consistency_kernel_left", &err);
    //CHECK_OCL_ERROR(err, "Create check_consistency_kernel_left");
    //
    //m_median_3x3 = clCreateKernel(sgm_program, "median3x3", &err);
    //CHECK_OCL_ERROR(err, "Create median3x3");
    //
    //m_copy_u8_to_u16 = clCreateKernel(sgm_program, "copy_u8_to_u16", &err);
    //CHECK_OCL_ERROR(err, "Create copy_u8_to_u16");
    //
    //m_clear_buffer = clCreateKernel(sgm_program, "clear_buffer", &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //d_src_left = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //d_src_left = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //d_src_right = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //d_left_census_cost = clCreateBuffer(m_cl_ctx,
    //    CL_MEM_READ_WRITE,
    //    sizeof(feature_type) * m_width * m_height,
    //    nullptr,
    //    &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //d_right_census_cost = clCreateBuffer(m_cl_ctx,
    //    CL_MEM_READ_WRITE,
    //    sizeof(feature_type) * m_width * m_height,
    //    nullptr,
    //    &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //const unsigned int num_paths = 4;// path_type == PathType::SCAN_4PATH ? 4 : 8;
    //const size_t buffer_size = m_width * m_height * m_max_disparity * num_paths;
    //const size_t buffer_step = m_width * m_height * m_max_disparity;
    //// cost type is uint8_t
    //d_cost_buffer = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //for (int i = 0; i < 8; ++i)
    //{
    //    cl_buffer_region region = { buffer_step * i, buffer_step };
    //    d_sub_buffers[i] = clCreateSubBuffer(d_cost_buffer,
    //        CL_MEM_READ_WRITE,
    //        CL_BUFFER_CREATE_TYPE_REGION,
    //       &region, &err);
    //}
    //
    //d_left_disparity = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //d_right_disparity = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //d_tmp_left_disp = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //d_tmp_right_disp = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    //CHECK_OCL_ERROR(err, "Create clear_buffer");
    //
    //
    ////setup kernels
    //err = clSetKernelArg(m_census_transform_kernel, 0, sizeof(cl_mem), &d_left_census_cost);
    //err = clSetKernelArg(m_census_transform_kernel, 1, sizeof(cl_mem), &d_src_left);
    //err = clSetKernelArg(m_census_transform_kernel, 2, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_census_transform_kernel, 3, sizeof(m_height), &m_height);
    //err = clSetKernelArg(m_census_transform_kernel, 4, sizeof(m_width), &m_width);
    //CHECK_OCL_ERROR(err, "error settings parameters");
    //
    ////
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 0, sizeof(cl_mem), &d_sub_buffers[0]);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 1, sizeof(cl_mem), &d_left_census_cost);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 2, sizeof(cl_mem), &d_right_census_cost);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 3, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 4, sizeof(m_height), &m_height);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 5, sizeof(m_p1), &m_p1);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 6, sizeof(m_p2), &m_p2);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 7, sizeof(m_min_disp), &m_min_disp);
    //CHECK_OCL_ERROR(err, "error settings parameters");
    //
    ////
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 0, sizeof(cl_mem), &d_sub_buffers[1]);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 1, sizeof(cl_mem), &d_left_census_cost);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 2, sizeof(cl_mem), &d_right_census_cost);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 3, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 4, sizeof(m_height), &m_height);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 5, sizeof(m_p1), &m_p1);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 6, sizeof(m_p2), &m_p2);
    //err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 7, sizeof(m_min_disp), &m_min_disp);
    //CHECK_OCL_ERROR(err, "error settings parameters");



    ////m_matching_cost_kernel_128->setArgs(d_left, d_right, d_matching_cost, m_width, m_height);
    //clSetKernelArg(m_matching_cost_kernel_128, 0, sizeof(cl_mem), &d_left);
    //clSetKernelArg(m_matching_cost_kernel_128, 1, sizeof(cl_mem), &d_right);
    //clSetKernelArg(m_matching_cost_kernel_128, 2, sizeof(cl_mem), &d_matching_cost);
    //clSetKernelArg(m_matching_cost_kernel_128, 3, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_matching_cost_kernel_128, 4, sizeof(m_height), &m_height);
    //CHECK_OCL_ERROR(err, "error settings parameters");
    //
    //
    //auto setOptDirKernelsArgs = [&](cl_kernel kernel)
    //{
    //    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matching_cost);
    //    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_scost);
    //    clSetKernelArg(kernel, 2, sizeof(m_width), &m_width);
    //    err = clSetKernelArg(kernel, 3, sizeof(m_height), &m_height);
    //    CHECK_OCL_ERROR(err, "error settings parameters");
    //};
    //
    //
    ////m_compute_stereo_horizontal_dir_kernel_0->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_horizontal_dir_kernel_0);
    //
    ////m_compute_stereo_horizontal_dir_kernel_4->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_horizontal_dir_kernel_4);
    //
    ////m_compute_stereo_vertical_dir_kernel_2->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_vertical_dir_kernel_2);
    //
    ////m_compute_stereo_vertical_dir_kernel_6->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_vertical_dir_kernel_6);
    ////
    ////m_compute_stereo_oblique_dir_kernel_1->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_oblique_dir_kernel_1);
    //
    ////m_compute_stereo_oblique_dir_kernel_3->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_oblique_dir_kernel_3);
    //
    ////m_compute_stereo_oblique_dir_kernel_5->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_oblique_dir_kernel_5);
    //
    ////m_compute_stereo_oblique_dir_kernel_7->setArgs(d_matching_cost, d_scost, m_width, m_height);
    //setOptDirKernelsArgs(m_compute_stereo_oblique_dir_kernel_7);
    //
    //
    ////m_winner_takes_all_kernel128->setArgs(d_left_disparity, d_right_disparity, d_scost, m_width, m_height);
    //clSetKernelArg(m_winner_takes_all_kernel128, 0, sizeof(cl_mem), &d_left_disparity);
    //clSetKernelArg(m_winner_takes_all_kernel128, 1, sizeof(cl_mem), &d_right_disparity);
    //clSetKernelArg(m_winner_takes_all_kernel128, 2, sizeof(cl_mem), &d_scost);
    //clSetKernelArg(m_winner_takes_all_kernel128, 3, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_winner_takes_all_kernel128, 4, sizeof(m_height), &m_height);
    //CHECK_OCL_ERROR(err, "error settings parameters");
    //
    ////m_check_consistency_left->setArgs(d_tmp_left_disp, d_tmp_right_disp, d_src_left, m_width, m_height);
    //clSetKernelArg(m_check_consistency_left, 0, sizeof(cl_mem), &d_tmp_left_disp);
    //clSetKernelArg(m_check_consistency_left, 1, sizeof(cl_mem), &d_tmp_right_disp);
    //clSetKernelArg(m_check_consistency_left, 2, sizeof(cl_mem), &d_src_left);
    //clSetKernelArg(m_check_consistency_left, 3, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_check_consistency_left, 4, sizeof(m_height), &m_height);
    //CHECK_OCL_ERROR(err, "error settings parameters");
    //
    ////m_median_3x3->setArgs(d_left_disparity, d_tmp_left_disp, m_width, m_height);
    //clSetKernelArg(m_median_3x3, 0, sizeof(cl_mem), &d_left_disparity);
    //clSetKernelArg(m_median_3x3, 1, sizeof(cl_mem), &d_tmp_left_disp);
    //clSetKernelArg(m_median_3x3, 2, sizeof(m_width), &m_width);
    //err = clSetKernelArg(m_median_3x3, 3, sizeof(m_height), &m_height);
    //CHECK_OCL_ERROR(err, "error settings parameters");
    //
    ////todo check boundary
    ////m_copy_u8_to_u16->setArgs(d_matching_cost, d_scost);
    //clSetKernelArg(m_copy_u8_to_u16, 0, sizeof(cl_mem), &d_matching_cost);
    //clSetKernelArg(m_copy_u8_to_u16, 1, sizeof(cl_mem), &d_scost);
}

template <typename input_type>
void StereoSGM<input_type>::finishQueue()
{
    cl_int err = clFinish(m_cl_cmd_queue);
    CHECK_OCL_ERROR(err, "Error finishing queue");
}


//void StereoSGM::mem_init()
//{
    //{
    //    clSetKernelArg(m_clear_buffer, 0, sizeof(cl_mem), &d_left_disparity);
    //    size_t global_size = (m_width* m_height * sizeof(uint16_t) / 32 / 256) * 256;
    //    size_t local_size = 256;
    //
    //    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
    //        m_clear_buffer,
    //        1,
    //        nullptr,
    //        &global_size,
    //        &local_size,
    //        0, nullptr, nullptr);
    //}
    //{
    //    clSetKernelArg(m_clear_buffer, 0, sizeof(cl_mem), &d_right_disparity);
    //    size_t global_size = (m_width * m_height * sizeof(uint16_t) / 32 / 256) * 256;
    //    size_t local_size = 256;
    //    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
    //        m_clear_buffer,
    //        1,
    //        nullptr,
    //        &global_size,
    //        &local_size,
    //        0, nullptr, nullptr);
    //}
    ////{
    ////    clSetKernelArg(m_clear_buffer, 0, sizeof(cl_mem), &d_scost);
    ////    size_t global_size = (m_width * m_height * sizeof(uint16_t) * m_max_disparity / 32 / 256) * 256;
    ////    size_t local_size = 256;
    ////    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
    ////        m_clear_buffer,
    ////        1,
    ////        nullptr,
    ////        &global_size,
    ////        &local_size,
    ////        0, nullptr, nullptr);
    ////
    ////}
//}

//void StereoSGM::path_aggregation()
//{
//
//    static constexpr unsigned int WARP_SIZE = 32u;
//    static constexpr unsigned int DP_BLOCK_SIZE = 16u;
//    static constexpr unsigned int BLOCK_SIZE_PA = WARP_SIZE * 8u;
//
//    static const unsigned int SUBGROUP_SIZE = m_max_disparity / DP_BLOCK_SIZE;
//    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE_PA / SUBGROUP_SIZE;
//
//    //vertical directions
//    //up down dir
//    const size_t gdim = (m_width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
//    const size_t bdim = BLOCK_SIZE_PA;
//
//    size_t global_size[1] = {gdim * bdim};
//    size_t local_size[1] = { bdim};
//
//
//    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
//        m_aggregate_vertical_path_kernel_dir_1,
//        1,
//        nullptr,
//        global_size,
//        local_size,
//        0, nullptr, nullptr);
//    CHECK_OCL_ERROR(err, "Error finishing queue");
//    finishQueue();
//    err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
//        m_aggregate_vertical_path_kernel_dir__1,
//        1,
//        nullptr,
//        global_size,
//        local_size,
//        0, nullptr, nullptr);
//    CHECK_OCL_ERROR(err, "Error finishing queue");
//    finishQueue();
//
//
//    int alma = 0;
//}

//void StereoSGM::winner_takes_all()
//{
//    const size_t WTA_PIXEL_IN_BLOCK = 8;
//    //(*m_winner_takes_all_kernel128)(0,
//    //    napalm::ImgRegion(m_width / WTA_PIXEL_IN_BLOCK, 1 * m_height),
//    //    napalm::ImgRegion(32, WTA_PIXEL_IN_BLOCK));
//
//    size_t global_size[2] = {
//        (m_width / WTA_PIXEL_IN_BLOCK) * 32,
//        (1 * m_height) * WTA_PIXEL_IN_BLOCK
//    };
//    size_t local_size[2] = { 32, WTA_PIXEL_IN_BLOCK };
//    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
//        m_winner_takes_all_kernel128,
//        2,
//        nullptr,
//        global_size,
//        local_size,
//        0, nullptr, nullptr);
//}

//void StereoSGM::median()
//{
//    size_t global_size[2] = {
//        (size_t)((m_width + 16 - 1) / 16) * 16,
//        (size_t)((m_height + 16 - 1) / 16) * 16
//    };
//    size_t local_size[2] = { 16, 16 };
//
//    //m_median_3x3->setArgs(d_left_disparity, d_tmp_left_disp);
//    //(*m_median_3x3)(0,
//    //    napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
//    //    napalm::ImgRegion(16, 16));
//
//    clSetKernelArg(m_median_3x3, 0, sizeof(cl_mem), &d_left_disparity);
//    clSetKernelArg(m_median_3x3, 1, sizeof(cl_mem), &d_tmp_left_disp);
//    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
//        m_median_3x3,
//        2,
//        nullptr,
//        global_size,
//        local_size,
//        0, nullptr, nullptr);
//
//    //m_median_3x3->setArgs(d_right_disparity, d_tmp_right_disp);
//    //(*m_median_3x3)(0,
//    //    napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
//    //    napalm::ImgRegion(16, 16));
//    clSetKernelArg(m_median_3x3, 0, sizeof(cl_mem), &d_right_disparity);
//    clSetKernelArg(m_median_3x3, 1, sizeof(cl_mem), &d_tmp_right_disp);
//    err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
//        m_median_3x3,
//        2,
//        nullptr,
//        global_size,
//        local_size,
//        0, nullptr, nullptr);
//
//
//}

//void StereoSGM::check_consistency_left()
//{
//    //(*m_check_consistency_left)(0,
//    //    napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
//    //    napalm::ImgRegion(16, 16));
//
//    size_t global_size[2] = {
//        (size_t)((m_width + 16 - 1) / 16) * 16,
//        (size_t)((m_height + 16 - 1) / 16) * 16
//    };
//    size_t local_size[2] = { 16, 16 };
//    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
//        m_check_consistency_left,
//        2,
//        nullptr,
//        global_size,
//        local_size,
//        0, nullptr, nullptr);
//}

// explicit instantiate for uint8_t and uint16_t
template class StereoSGM<uint8_t>;
template class StereoSGM<uint16_t>;

}
}
