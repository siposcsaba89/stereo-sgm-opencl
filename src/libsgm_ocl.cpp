#include "libsgm_ocl/libsgm_ocl.h"
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
    std::cout << "opencl error : " << errinfo << std::endl;
}

StereoSGMCL::StereoSGMCL(int width,
    int height,
    int max_disp_size,
    int platform_idx,
    int device_idx)
    :
    m_width(width),
    m_height(height),
    m_max_disparity(max_disp_size)
{
    initCLCTX(platform_idx, device_idx);
    initCL();
}

StereoSGMCL::StereoSGMCL(int width, int height, int max_disp_size, cl_context ctx)
    :
    m_width(width),
    m_height(height),
    m_max_disparity(max_disp_size),
    m_cl_ctx(ctx)
{
    initCL();
}


StereoSGMCL::~StereoSGMCL()
{
    //delete d_src_left;
    clReleaseMemObject(d_src_left);
    //delete d_src_right;
    clReleaseMemObject(d_src_right);
    //delete d_left;
    clReleaseMemObject(d_left_census_cost);
    //delete d_right;
    clReleaseMemObject(d_right_census_cost);
    //delete d_matching_cost;
    clReleaseMemObject(d_cost_buffer);
    //delete d_left_disparity;
    clReleaseMemObject(d_left_disparity);
    //delete d_right_disparity;
    clReleaseMemObject(d_right_disparity);
    //delete d_tmp_left_disp;
    clReleaseMemObject(d_tmp_left_disp);
    //delete d_tmp_right_disp;
    clReleaseMemObject(d_tmp_right_disp);
}

bool StereoSGMCL::init(int platform_idx, int device_idx)
{
    initCLCTX(platform_idx, device_idx);
    initCL();
    return true;
}

void StereoSGMCL::execute(void* left_data, void* right_data, void* output_buffer)
{
    cl_int err;

    //d_src_left->write(left_data);
    err = clEnqueueWriteBuffer(m_cl_cmd_queue,
        d_src_left,
        false, // blocking
        0, //offset
        m_width * m_height,
        left_data,
        0, nullptr, nullptr);

    //d_src_right->write(right_data);
    err = clEnqueueWriteBuffer(m_cl_cmd_queue,
        d_src_right,
        false, // blocking
        0, //offset
        m_width * m_height,
        right_data,
        0, nullptr, nullptr);

    census();

    mem_init();
    //m_context->finish(0);
    //m_context->finish(0);


    //(*m_copy_u8_to_u16)(0,
    //	m_width * m_height,
    //	128);
    //m_context->finish(0);

    path_aggregation();
    finishQueue();


    winner_takes_all();
    //m_context->finish(0);

    median();
    //m_context->finish(0);

    check_consistency_left();
    finishQueue();

    err = clEnqueueReadBuffer(m_cl_cmd_queue,
        d_tmp_left_disp,
        true, // blocking
        0, //offset
        m_width * m_height * sizeof(uint16_t),
        output_buffer,
        0, nullptr, nullptr);

    //d_tmp_left_disp->read(output_buffer);
    //m_context->finish(0);

}

#define CHECK_OCL_ERROR(err, msg) \
    if (err != CL_SUCCESS) \
    { \
        std::cout << "OCL_ERROR at line " << __LINE__ << ". Message: " << msg << std::endl; \
    }\

void StereoSGMCL::initCL()
{
    cl_int err;
    m_cl_cmd_queue = clCreateCommandQueue(m_cl_ctx, m_cl_device, 0, &err);
    CHECK_OCL_ERROR(err, "Failed to create command queue");

    std::ifstream t("d:/Projects/stereo-sgm-opencl/libsgm_ocl/sgm.cl");
    std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    size_t data_length = str.size();
    const char* data = str.c_str();
    cl_program sgm_program = clCreateProgramWithSource(m_cl_ctx, 1, &data, &data_length, &err);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("Cannot create ocl program!");
    }

    err = clBuildProgram(sgm_program, 1, &m_cl_device, nullptr, nullptr, nullptr);

    size_t build_log_size = 0;
    clGetProgramBuildInfo(sgm_program, m_cl_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size);
    std::string build_log;
    build_log.resize(build_log_size);
    clGetProgramBuildInfo(sgm_program, m_cl_device, CL_PROGRAM_BUILD_LOG,
        build_log_size,
        &build_log[0],
        nullptr);
    std::cout << "OpenCL build info: " << build_log << std::endl;

    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("Cannot build ocl program!");
    }


    m_census_transform_kernel = clCreateKernel(sgm_program, "census_transform_kernel", &err);
    CHECK_OCL_ERROR(err, "Create census_kernel");
    //path aggregation kernels
    m_aggregate_vertical_path_kernel_dir_1 = clCreateKernel(sgm_program, "aggregate_vertical_path_kernel", &err);
    CHECK_OCL_ERROR(err, "Create aggregate_vertical_path_kernel");
    m_aggregate_vertical_path_kernel_dir__1 = clCreateKernel(sgm_program, "aggregate_vertical_path_kernel_down2up", &err);
    CHECK_OCL_ERROR(err, "Create aggregate_vertical_path_kernel_down2up");
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
    m_clear_buffer = clCreateKernel(sgm_program, "clear_buffer", &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    d_src_left = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    d_src_left = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");
    d_src_right = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    d_left_census_cost = clCreateBuffer(m_cl_ctx,
        CL_MEM_READ_WRITE,
        sizeof(feature_type) * m_width * m_height,
        nullptr,
        &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");
    d_right_census_cost = clCreateBuffer(m_cl_ctx,
        CL_MEM_READ_WRITE,
        sizeof(feature_type) * m_width * m_height,
        nullptr,
        &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    const unsigned int num_paths = 4;// path_type == PathType::SCAN_4PATH ? 4 : 8;
    const size_t buffer_size = m_width * m_height * m_max_disparity * num_paths;
    const size_t buffer_step = m_width * m_height * m_max_disparity;
    // cost type is uint8_t
    d_cost_buffer = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    for (int i = 0; i < 8; ++i)
    {
        cl_buffer_region region = { buffer_step * i, buffer_step };
        d_sub_buffers[i] = clCreateSubBuffer(d_cost_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
           &region, &err);
    }

    d_left_disparity = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    d_right_disparity = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    d_tmp_left_disp = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");

    d_tmp_right_disp = clCreateBuffer(m_cl_ctx, CL_MEM_READ_WRITE, sizeof(uint16_t) * m_width * m_height, nullptr, &err);
    CHECK_OCL_ERROR(err, "Create clear_buffer");


    //setup kernels
    err = clSetKernelArg(m_census_transform_kernel, 0, sizeof(cl_mem), &d_left_census_cost);
    err = clSetKernelArg(m_census_transform_kernel, 1, sizeof(cl_mem), &d_src_left);
    err = clSetKernelArg(m_census_transform_kernel, 2, sizeof(m_width), &m_width);
    err = clSetKernelArg(m_census_transform_kernel, 3, sizeof(m_height), &m_height);
    err = clSetKernelArg(m_census_transform_kernel, 4, sizeof(m_width), &m_width);
    CHECK_OCL_ERROR(err, "error settings parameters");

    //
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 0, sizeof(cl_mem), &d_sub_buffers[0]);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 1, sizeof(cl_mem), &d_left_census_cost);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 2, sizeof(cl_mem), &d_right_census_cost);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 3, sizeof(m_width), &m_width);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 4, sizeof(m_height), &m_height);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 5, sizeof(m_p1), &m_p1);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 6, sizeof(m_p2), &m_p2);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir_1, 7, sizeof(m_min_disp), &m_min_disp);
    CHECK_OCL_ERROR(err, "error settings parameters");

    //
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 0, sizeof(cl_mem), &d_sub_buffers[1]);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 1, sizeof(cl_mem), &d_left_census_cost);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 2, sizeof(cl_mem), &d_right_census_cost);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 3, sizeof(m_width), &m_width);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 4, sizeof(m_height), &m_height);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 5, sizeof(m_p1), &m_p1);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 6, sizeof(m_p2), &m_p2);
    err = clSetKernelArg(m_aggregate_vertical_path_kernel_dir__1, 7, sizeof(m_min_disp), &m_min_disp);
    CHECK_OCL_ERROR(err, "error settings parameters");



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

void StereoSGMCL::initCLCTX(int platform_idx, int device_idx)
{
    cl_uint num_platform;
    clGetPlatformIDs(0, nullptr, &num_platform);
    assert((size_t)platform_idx < num_platform);
    std::vector<cl_platform_id> platform_ids(num_platform);
    clGetPlatformIDs(num_platform, platform_ids.data(), nullptr);
    cl_uint num_devices;
    clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    assert((size_t)device_idx < num_devices);
    std::vector<cl_device_id> cl_devices(num_devices);
    clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, num_devices, cl_devices.data(), nullptr);
    m_cl_device = cl_devices[device_idx];
    cl_int err;
    m_cl_ctx = clCreateContext(nullptr, 1, &cl_devices[device_idx], context_error_callback, NULL, &err);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error creating context " << err << std::endl;
        throw std::runtime_error("Error creating context!");
    }
    {
        size_t name_size_in_bytes;
        clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_NAME, 0, nullptr, &name_size_in_bytes);
        std::string platform_name;
        platform_name.resize(name_size_in_bytes);
        clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_NAME,
            platform_name.size(),
            (void*)platform_name.data(), nullptr);
        std::cout << "Platform name: " << platform_name << std::endl;
    }
    {
        size_t name_size_in_bytes;
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME, 0, nullptr, &name_size_in_bytes);
        std::string dev_name;
        dev_name.resize(name_size_in_bytes);
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME,
            dev_name.size(),
            (void*)dev_name.data(), nullptr);
        std::cout << "Device name: " << dev_name << std::endl;
    }
}

void StereoSGMCL::finishQueue()
{
    cl_int err = clFinish(m_cl_cmd_queue);
    CHECK_OCL_ERROR(err, "Error finishing queue");
}


#define WINDOW_WIDTH  9
#define WINDOW_HEIGHT  7
#define BLOCK_SIZE 128
#define LINES_PER_BLOCK 16
void StereoSGMCL::census()
{
    const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
    const int height_per_block = LINES_PER_BLOCK;

    //setup kernels
    size_t global_size[2] = {
        (size_t)((m_width + width_per_block - 1) / width_per_block * BLOCK_SIZE),
        (size_t)((m_height + height_per_block - 1) / height_per_block) 
    };
    size_t local_size[2] = { BLOCK_SIZE, 1 };
    clSetKernelArg(m_census_transform_kernel, 0, sizeof(cl_mem), &d_left_census_cost);
    clSetKernelArg(m_census_transform_kernel, 1, sizeof(cl_mem), &d_src_left);
    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_census_transform_kernel,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing census kernel");
    clSetKernelArg(m_census_transform_kernel, 0, sizeof(cl_mem), &d_right_census_cost);
    clSetKernelArg(m_census_transform_kernel, 1, sizeof(cl_mem), &d_src_right);
    err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_census_transform_kernel,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error enequeuing census kernel");
    //TODO not necessary finish
    finishQueue();
}

void StereoSGMCL::mem_init()
{
    {
        clSetKernelArg(m_clear_buffer, 0, sizeof(cl_mem), &d_left_disparity);
        size_t global_size = (m_width* m_height * sizeof(uint16_t) / 32 / 256) * 256;
        size_t local_size = 256;

        cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
            m_clear_buffer,
            1,
            nullptr,
            &global_size,
            &local_size,
            0, nullptr, nullptr);
    }
    {
        clSetKernelArg(m_clear_buffer, 0, sizeof(cl_mem), &d_right_disparity);
        size_t global_size = (m_width * m_height * sizeof(uint16_t) / 32 / 256) * 256;
        size_t local_size = 256;
        cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
            m_clear_buffer,
            1,
            nullptr,
            &global_size,
            &local_size,
            0, nullptr, nullptr);
    }
    //{
    //    clSetKernelArg(m_clear_buffer, 0, sizeof(cl_mem), &d_scost);
    //    size_t global_size = (m_width * m_height * sizeof(uint16_t) * m_max_disparity / 32 / 256) * 256;
    //    size_t local_size = 256;
    //    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
    //        m_clear_buffer,
    //        1,
    //        nullptr,
    //        &global_size,
    //        &local_size,
    //        0, nullptr, nullptr);
    //
    //}
}

void StereoSGMCL::path_aggregation()
{

    static constexpr unsigned int WARP_SIZE = 32u;
    static constexpr unsigned int DP_BLOCK_SIZE = 16u;
    static constexpr unsigned int BLOCK_SIZE_PA = WARP_SIZE * 8u;

    static const unsigned int SUBGROUP_SIZE = m_max_disparity / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE_PA / SUBGROUP_SIZE;

    //vertical directions
    //up down dir
    const size_t gdim = (m_width + PATHS_PER_BLOCK - 1) / PATHS_PER_BLOCK;
    const size_t bdim = BLOCK_SIZE_PA;

    size_t global_size[1] = {gdim * bdim};
    size_t local_size[1] = { bdim};


    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_aggregate_vertical_path_kernel_dir_1,
        1,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error finishing queue");
    finishQueue();
    err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_aggregate_vertical_path_kernel_dir__1,
        1,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
    CHECK_OCL_ERROR(err, "Error finishing queue");
    finishQueue();


    int alma = 0;
}

void StereoSGMCL::winner_takes_all()
{
    const size_t WTA_PIXEL_IN_BLOCK = 8;
    //(*m_winner_takes_all_kernel128)(0,
    //    napalm::ImgRegion(m_width / WTA_PIXEL_IN_BLOCK, 1 * m_height),
    //    napalm::ImgRegion(32, WTA_PIXEL_IN_BLOCK));

    size_t global_size[2] = {
        (m_width / WTA_PIXEL_IN_BLOCK) * 32,
        (1 * m_height) * WTA_PIXEL_IN_BLOCK
    };
    size_t local_size[2] = { 32, WTA_PIXEL_IN_BLOCK };
    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_winner_takes_all_kernel128,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
}

void StereoSGMCL::median()
{
    size_t global_size[2] = {
        (size_t)((m_width + 16 - 1) / 16) * 16,
        (size_t)((m_height + 16 - 1) / 16) * 16
    };
    size_t local_size[2] = { 16, 16 };

    //m_median_3x3->setArgs(d_left_disparity, d_tmp_left_disp);
    //(*m_median_3x3)(0,
    //    napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
    //    napalm::ImgRegion(16, 16));

    clSetKernelArg(m_median_3x3, 0, sizeof(cl_mem), &d_left_disparity);
    clSetKernelArg(m_median_3x3, 1, sizeof(cl_mem), &d_tmp_left_disp);
    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_median_3x3,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);

    //m_median_3x3->setArgs(d_right_disparity, d_tmp_right_disp);
    //(*m_median_3x3)(0,
    //    napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
    //    napalm::ImgRegion(16, 16));
    clSetKernelArg(m_median_3x3, 0, sizeof(cl_mem), &d_right_disparity);
    clSetKernelArg(m_median_3x3, 1, sizeof(cl_mem), &d_tmp_right_disp);
    err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_median_3x3,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);


}

void StereoSGMCL::check_consistency_left()
{
    //(*m_check_consistency_left)(0,
    //    napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
    //    napalm::ImgRegion(16, 16));

    size_t global_size[2] = {
        (size_t)((m_width + 16 - 1) / 16) * 16,
        (size_t)((m_height + 16 - 1) / 16) * 16
    };
    size_t local_size[2] = { 16, 16 };
    cl_int err = clEnqueueNDRangeKernel(m_cl_cmd_queue,
        m_check_consistency_left,
        2,
        nullptr,
        global_size,
        local_size,
        0, nullptr, nullptr);
}
