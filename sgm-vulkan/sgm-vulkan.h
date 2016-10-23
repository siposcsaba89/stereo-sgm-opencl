#pragma once

#include <vulkan/vulkan.hpp>

class StereoSGMVULKAN
{
public:
	StereoSGMVULKAN(int width, int height, int max_disp_size);
	~StereoSGMVULKAN();
	bool init();
	void execute(void * left_data, void * right_data, void * output_buffer);
private:
	void initVulkan();
	void census();
	void mem_init();
	void matching_cost();
	void scan_cost();
	void winner_takes_all();
	void median();
	void check_consistency_left();
private:

	int m_width = -1;
	int m_height = -1;
	int m_max_disparity = -1;
    bool m_enable_validation = true;
    vk::Instance m_vk_instance;
    vk::PhysicalDevice m_vk_physical_device;
    vk::PhysicalDeviceFeatures m_vk_device_features; 
    vk::Device m_vk_device;
    vk::Queue m_vk_queue;

    PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallback = VK_NULL_HANDLE;
    PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallback = VK_NULL_HANDLE;
    PFN_vkDebugReportMessageEXT dbgBreakCallback = VK_NULL_HANDLE;
    VkDebugReportCallbackEXT msgCallback;
//	cl::Context m_context;
//	cl::CommandQueue m_command_queue;
//	cl::Program m_program;
//	cl::Device m_device;
//	//cl::Kernel 
//	//buffers
//
//
//
//	cl::Kernel m_census_kernel;
//	cl::Kernel m_matching_cost_kernel_128;
//
//	cl::Kernel m_compute_stereo_horizontal_dir_kernel_0;
//	cl::Kernel m_compute_stereo_horizontal_dir_kernel_4;
//	cl::Kernel m_compute_stereo_vertical_dir_kernel_2;
//	cl::Kernel m_compute_stereo_vertical_dir_kernel_6;
//
//	cl::Kernel m_compute_stereo_oblique_dir_kernel_1;
//	cl::Kernel m_compute_stereo_oblique_dir_kernel_3;
//	cl::Kernel m_compute_stereo_oblique_dir_kernel_5;
//	cl::Kernel m_compute_stereo_oblique_dir_kernel_7;
//
//
//	cl::Kernel m_winner_takes_all_kernel128;
//	
//	cl::Kernel m_check_consistency_left;
//
//	cl::Kernel m_median_3x3;
//
//	cl::Kernel m_copy_u8_to_u16;
//	cl::Kernel m_clear_buffer;
//
//
	vk::Buffer d_src_left, d_src_right, d_left, d_right, d_matching_cost,
		d_scost, d_left_disparity, d_right_disparity,
		d_tmp_left_disp, d_tmp_right_disp;


};