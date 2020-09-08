#pragma once

#include <CL/cl.h>
#include <inttypes.h>

// pixel and feature type defs
#define pixel_type uint8_t
#define feature_type uint32_t


class StereoSGMCL
{
public:
    StereoSGMCL(int width,
        int height,
        int max_disp_size,
        int platform_idx = 0,
        int device_idx = 0);
    StereoSGMCL(int width,
        int height,
        int max_disp_size,
        cl_context ctx);
    ~StereoSGMCL();
    bool init(int platform_idx, int device_idx);
    void execute(void* left_data, void* right_data, void* output_buffer);
private:
    void initCL();
    void initCLCTX(int platform_idx, int device_idx);
    void finishQueue();
    void census();
    void mem_init();
    void path_aggregation();
    void winner_takes_all();
    void median();
    void check_consistency_left();
private:
    cl_context m_cl_ctx;
    cl_device_id m_cl_device;
    cl_command_queue m_cl_cmd_queue;

    int m_width = -1;
    int m_height = -1;
    int m_max_disparity = -1;
    uint32_t m_p1 = 10;
    uint32_t m_p2 = 120;
    int32_t m_min_disp = 0;


    cl_kernel m_census_transform_kernel;
    cl_kernel m_aggregate_vertical_path_kernel_dir_1;
    cl_kernel m_aggregate_vertical_path_kernel_dir__1;

    //cl_kernel m_compute_stereo_horizontal_dir_kernel_0;
    //cl_kernel m_compute_stereo_horizontal_dir_kernel_4;
    //cl_kernel m_compute_stereo_vertical_dir_kernel_2;
    //cl_kernel m_compute_stereo_vertical_dir_kernel_6;
    //
    //cl_kernel m_compute_stereo_oblique_dir_kernel_1;
    //cl_kernel m_compute_stereo_oblique_dir_kernel_3;
    //cl_kernel m_compute_stereo_oblique_dir_kernel_5;
    //cl_kernel m_compute_stereo_oblique_dir_kernel_7;


    cl_kernel m_winner_takes_all_kernel128;

    cl_kernel m_check_consistency_left;

    cl_kernel m_median_3x3;
    cl_kernel m_copy_u8_to_u16;
    cl_kernel m_clear_buffer;


    cl_mem d_src_left, d_src_right,
        d_left_census_cost, d_right_census_cost,
        d_cost_buffer,
        d_left_disparity, d_right_disparity,
        d_tmp_left_disp, d_tmp_right_disp;

    cl_mem d_sub_buffers[8];
};