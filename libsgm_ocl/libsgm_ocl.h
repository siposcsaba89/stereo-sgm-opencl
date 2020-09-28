#pragma once

#include <CL/cl.h>
#include <inttypes.h>

// pixel and feature type defs
#define pixel_type uint8_t
#define feature_type uint32_t


namespace sgm
{
namespace cl
{

//types.hpp
using feature_type = uint32_t;
using cost_type = uint8_t;
using cost_sum_type = uint16_t;
using output_type = uint16_t;



/**
* @enum DST_TYPE
* Indicates input/output pointer type.
*/
enum EXECUTE_INOUT {
    EXECUTE_INOUT_HOST2HOST = (0 << 1) | 0,
    EXECUTE_INOUT_HOST2DEVICE = (1 << 1) | 0,
    EXECUTE_INOUT_DEVICE2HOST = (0 << 1) | 1,
    EXECUTE_INOUT_DEVICE2DEVICE = (1 << 1) | 1,
};

/**
 Indicates number of scanlines which will be used.
*/
enum class PathType {
    SCAN_4PATH, //>! Horizontal and vertical paths.
    SCAN_8PATH  //>! Horizontal, vertical and oblique paths.
};


struct Parameters
{
    int P1;
    int P2;
    float uniqueness;
    bool subpixel;
    PathType path_type;
    int min_disp;
    int LR_max_diff;

    /**
    * @param P1 Penalty on the disparity change by plus or minus 1 between nieghbor pixels.
    * @param P2 Penalty on the disparity change by more than 1 between neighbor pixels.
    * @param uniqueness Margin in ratio by which the best cost function value should be at least second one.
    * @param subpixel Disparity value has 4 fractional bits if subpixel option is enabled.
    * @param path_type Number of scanlines used in cost aggregation.
    * @param min_disp Minimum possible disparity value.
    * @param LR_max_diff Acceptable difference pixels which is used in LR check consistency. LR check consistency will be disabled if this value is set to negative.
    */
    Parameters(int P1 = 10, int P2 = 120, float uniqueness = 0.95f, bool subpixel = false, PathType path_type = PathType::SCAN_8PATH, int min_disp = 0, int LR_max_diff = 1)
        : P1(P1)
        , P2(P2)
        , uniqueness(uniqueness)
        , subpixel(subpixel)
        , path_type(path_type)
        , min_disp(min_disp)
        , LR_max_diff(LR_max_diff)
    { }
};

class StereoSGM
{
public:

    /**
    * @param width Processed image's width.
    * @param height Processed image's height.
    * @param disparity_size It must be 64, 128 or 256.
    * @param input_depth_bits Processed image's bits per pixel. It must be 8 or 16.
    * @param output_depth_bits Disparity image's bits per pixel. It must be 8 or 16.
    * @param inout_type Specify input/output pointer type. See sgm::EXECUTE_TYPE.
    * @attention
    * output_depth_bits must be set to 16 when subpixel is enabled.
    */
    StereoSGM(int width,
        int height,
        int disparity_size,
        int input_depth_bits,
        int output_depth_bits,
        EXECUTE_INOUT inout_type,
        cl_context ctx,
        const Parameters& param = Parameters());

    /**
    * @param width Processed image's width.
    * @param height Processed image's height.
    * @param disparity_size It must be 64, 128 or 256.
    * @param input_depth_bits Processed image's bits per pixel. It must be 8 or 16.
    * @param output_depth_bits Disparity image's bits per pixel. It must be 8 or 16.
    * @param inout_type Specify input/output pointer type. See sgm::EXECUTE_TYPE.
    * @attention
    * output_depth_bits must be set to 16 when subpixel is enabled.
    */
    StereoSGM(int width,
        int height,
        int disparity_size,
        int input_depth_bits,
        int output_depth_bits,
        EXECUTE_INOUT inout_type,
        cl_context ctx,
        const Parameters& param = Parameters());

    /**
    * @param width Processed image's width.
    * @param height Processed image's height.
    * @param disparity_size It must be 64, 128 or 256.
    * @param input_depth_bits Processed image's bits per pixel. It must be 8 or 16.
    * @param output_depth_bits Disparity image's bits per pixel. It must be 8 or 16.
    * @param src_pitch Source image's pitch (pixels).
    * @param dst_pitch Destination image's pitch (pixels).
    * @param inout_type Specify input/output pointer type. See sgm::EXECUTE_TYPE.
    * @attention
    * output_depth_bits must be set to 16 when subpixel is enabled.
    */
    StereoSGM(int width,
        int height,
        int disparity_size,
        int input_depth_bits,
        int output_depth_bits,
        int src_pitch,
        int dst_pitch,
        EXECUTE_INOUT inout_type,
        cl_context ctx,
        const Parameters& param = Parameters());

    ~StereoSGM();
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
    
    Parameters m_params = Parameters();


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

}
}