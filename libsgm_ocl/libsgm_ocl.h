#pragma once

#include <CL/cl.h>

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
    void matching_cost();
    void scan_cost();
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

    cl_kernel m_census_kernel;
    cl_kernel m_matching_cost_kernel_128;

    cl_kernel m_compute_stereo_horizontal_dir_kernel_0;
    cl_kernel m_compute_stereo_horizontal_dir_kernel_4;
    cl_kernel m_compute_stereo_vertical_dir_kernel_2;
    cl_kernel m_compute_stereo_vertical_dir_kernel_6;

    cl_kernel m_compute_stereo_oblique_dir_kernel_1;
    cl_kernel m_compute_stereo_oblique_dir_kernel_3;
    cl_kernel m_compute_stereo_oblique_dir_kernel_5;
    cl_kernel m_compute_stereo_oblique_dir_kernel_7;


    cl_kernel m_winner_takes_all_kernel128;

    cl_kernel m_check_consistency_left;

    cl_kernel m_median_3x3;
    cl_kernel m_copy_u8_to_u16;
    cl_kernel m_clear_buffer;


    cl_mem d_src_left, d_src_right, d_left, d_right, d_matching_cost,
        d_scost, d_left_disparity, d_right_disparity,
        d_tmp_left_disp, d_tmp_right_disp;


};