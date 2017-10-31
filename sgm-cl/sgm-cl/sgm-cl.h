#pragma once
#include <napalm/napalm.h>

class StereoSGMCL
{
public:
	StereoSGMCL(int width, int height, int max_disp_size, napalm::Context * ctx = nullptr);
	~StereoSGMCL();
	bool init();
	void execute(void * left_data, void * right_data, void * output_buffer);
private:
	void initCL();
	void census();
	void mem_init();
	void matching_cost();
	void scan_cost();
	void winner_takes_all();
	void median();
	void check_consistency_left();
private:
	napalm::Context * m_context;

	int m_width = -1;
	int m_height = -1;
	int m_max_disparity = -1;


	napalm::Kernel * m_census_kernel;
	napalm::Kernel * m_matching_cost_kernel_128;

	napalm::Kernel * m_compute_stereo_horizontal_dir_kernel_0;
	napalm::Kernel * m_compute_stereo_horizontal_dir_kernel_4;
	napalm::Kernel * m_compute_stereo_vertical_dir_kernel_2;
	napalm::Kernel * m_compute_stereo_vertical_dir_kernel_6;

	napalm::Kernel * m_compute_stereo_oblique_dir_kernel_1;
	napalm::Kernel * m_compute_stereo_oblique_dir_kernel_3;
	napalm::Kernel * m_compute_stereo_oblique_dir_kernel_5;
	napalm::Kernel * m_compute_stereo_oblique_dir_kernel_7;


	napalm::Kernel * m_winner_takes_all_kernel128;
	
	napalm::Kernel * m_check_consistency_left;

	napalm::Kernel * m_median_3x3;

	napalm::Kernel * m_copy_u8_to_u16;
	napalm::Kernel * m_clear_buffer;


	napalm::Buffer * d_src_left,* d_src_right,* d_left, *d_right, *d_matching_cost,
		*d_scost,* d_left_disparity,* d_right_disparity,
		*d_tmp_left_disp, *d_tmp_right_disp;


};