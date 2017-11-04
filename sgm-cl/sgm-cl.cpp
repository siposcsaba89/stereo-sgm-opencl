#include "sgm-cl/sgm-cl.h"
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include "napalm/gen/programs/sgm.h"
#include <napalm/context_manager.h>

int platform_index = 1;
int device_index = 0;


StereoSGMCL::StereoSGMCL(int width, int height, int max_disp_size, napalm::Context * ctx):
	m_width(width), m_height(height), m_max_disparity(max_disp_size), m_context(ctx)
{
    initCL();
}


StereoSGMCL::~StereoSGMCL()
{
    delete d_src_left;
    delete d_src_right;
    delete d_left;
    delete d_right;
    delete d_matching_cost;
    delete d_scost;
    delete d_left_disparity;
    delete d_right_disparity;
    delete d_tmp_left_disp;
    delete d_tmp_right_disp;
}

bool StereoSGMCL::init()
{
	initCL();
	return true;
}

void StereoSGMCL::execute(void * left_data, void * right_data, void * output_buffer)
{
    d_src_left->write(left_data);
    d_src_right->write(right_data);
    
    census();
    
    mem_init();
    //m_context->finish(0);
    matching_cost();
    //m_context->finish(0);
    
    
    //(*m_copy_u8_to_u16)(0,
    //	m_width * m_height,
    //	128);
    //m_context->finish(0);
    
    scan_cost();
    
    winner_takes_all();
    //m_context->finish(0);
    
    median();
    //m_context->finish(0);
    
    check_consistency_left();
    m_context->finish(0);
    
    d_tmp_left_disp->read(output_buffer);
    //m_context->finish(0);
    
}

void StereoSGMCL::initCL()
{
    if (m_context == nullptr)
    {
        m_context = napalm::ContextManager::getContextManager().getDefault("OpenCL", platform_index, device_index);
    }

    napalm::gen::sgm pr(*napalm::ProgramStore::create(m_context));

    m_census_kernel = &pr("census_kernel");
    m_matching_cost_kernel_128 = &pr("matching_cost_kernel_128");


    m_compute_stereo_horizontal_dir_kernel_0 = &pr("compute_stereo_horizontal_dir_kernel_0");
    m_compute_stereo_horizontal_dir_kernel_4 = &pr("compute_stereo_horizontal_dir_kernel_4");
    m_compute_stereo_vertical_dir_kernel_2 = &pr("compute_stereo_vertical_dir_kernel_2");
    m_compute_stereo_vertical_dir_kernel_6 = &pr("compute_stereo_vertical_dir_kernel_6");
    m_compute_stereo_oblique_dir_kernel_1 = &pr("compute_stereo_oblique_dir_kernel_1");
    m_compute_stereo_oblique_dir_kernel_3 = &pr("compute_stereo_oblique_dir_kernel_3");
    m_compute_stereo_oblique_dir_kernel_5 = &pr("compute_stereo_oblique_dir_kernel_5");
    m_compute_stereo_oblique_dir_kernel_7 = &pr("compute_stereo_oblique_dir_kernel_7");


    m_winner_takes_all_kernel128 = &pr("winner_takes_all_kernel128");

    m_check_consistency_left = &pr("check_consistency_kernel_left");

    m_median_3x3 = &pr("median3x3");

    m_copy_u8_to_u16 = &pr("copy_u8_to_u16");

    m_clear_buffer = &pr("clear_buffer");


    d_src_left = m_context->createBuffer(m_width * m_height);
    d_src_right = m_context->createBuffer(m_width * m_height);

    d_left = m_context->createBuffer(sizeof(uint64_t) * m_width * m_height);
    d_right = m_context->createBuffer(sizeof(uint64_t) * m_width * m_height);

    d_matching_cost = m_context->createBuffer(m_width * m_height * m_max_disparity);

    d_scost = m_context->createBuffer(sizeof(uint16_t) * m_width * m_height * m_max_disparity);

    d_left_disparity = m_context->createBuffer(sizeof(uint16_t) * m_width * m_height);

    d_right_disparity = m_context->createBuffer(sizeof(uint16_t) * m_width * m_height);

    d_tmp_left_disp = m_context->createBuffer(sizeof(uint16_t) * m_width * m_height);

    d_tmp_right_disp = m_context->createBuffer(sizeof(uint16_t) * m_width * m_height);

    //setup kernels
    m_census_kernel->setArgs(d_src_left, d_left, m_width, m_height);

    m_matching_cost_kernel_128->setArgs(d_left, d_right, d_matching_cost, m_width, m_height);

    m_compute_stereo_horizontal_dir_kernel_0->setArgs(d_matching_cost, d_scost, m_width, m_height);

    m_compute_stereo_horizontal_dir_kernel_4->setArgs(d_matching_cost, d_scost, m_width, m_height);

    m_compute_stereo_vertical_dir_kernel_2->setArgs(d_matching_cost, d_scost, m_width, m_height);

    m_compute_stereo_vertical_dir_kernel_6->setArgs(d_matching_cost, d_scost, m_width, m_height);


    m_compute_stereo_oblique_dir_kernel_1->setArgs(d_matching_cost, d_scost, m_width, m_height);

    m_compute_stereo_oblique_dir_kernel_3->setArgs(d_matching_cost, d_scost, m_width, m_height);

    m_compute_stereo_oblique_dir_kernel_5->setArgs(d_matching_cost, d_scost, m_width, m_height);

    m_compute_stereo_oblique_dir_kernel_7->setArgs(d_matching_cost, d_scost, m_width, m_height);


    m_winner_takes_all_kernel128->setArgs(d_left_disparity, d_right_disparity, d_scost, m_width, m_height);


    m_check_consistency_left->setArgs(d_tmp_left_disp, d_tmp_right_disp, d_src_left, m_width, m_height);

    m_median_3x3->setArgs(d_left_disparity, d_tmp_left_disp, m_width, m_height);

    m_copy_u8_to_u16->setArgs(d_matching_cost, d_scost);
}

void StereoSGMCL::census()
{
    //setup kernels
    m_census_kernel->setArgs(d_src_left, d_left);
    (*m_census_kernel)(0, napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
        napalm::ImgRegion(16, 16));
    m_context->finish(0);
    m_census_kernel->setArgs(d_src_right, d_right);
    (*m_census_kernel)(0, napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
        napalm::ImgRegion(16, 16));
    m_context->finish(0);
}

void StereoSGMCL::mem_init()
{
	//cl_int m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_left_disparity, 0, 0,
	//	m_width * m_height * sizeof(uint16_t));
	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
	//m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_right_disparity, 0, 0,
	//	m_width * m_height * sizeof(uint16_t));
	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
	//
	//m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_scost, 0, 0,
	//	m_width * m_height * sizeof(uint16_t) * m_max_disparity);
	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

    m_clear_buffer->setArgs(d_left_disparity);
    (*m_clear_buffer)(0, napalm::ImgRegion(m_width * m_height * sizeof(uint16_t) / 32 / 256),
        napalm::ImgRegion(256));

    m_clear_buffer->setArgs(d_right_disparity);
    (*m_clear_buffer)(0, napalm::ImgRegion(m_width * m_height * sizeof(uint16_t) / 32 / 256),
        napalm::ImgRegion(256));

    m_clear_buffer->setArgs(d_scost);
    (*m_clear_buffer)(0, napalm::ImgRegion(m_width * m_height * sizeof(uint16_t) * m_max_disparity / 32 / 256),
        napalm::ImgRegion(256));

}

void StereoSGMCL::matching_cost()
{
    int MCOST_LINES128 = 2;
    (*m_matching_cost_kernel_128)(0,
        napalm::ImgRegion(m_height / MCOST_LINES128, 1),
        napalm::ImgRegion(128, MCOST_LINES128));
}

void StereoSGMCL::scan_cost()
{
    //census_event.wait();
    static const int PATHS_IN_BLOCK = 8;
    (*m_compute_stereo_horizontal_dir_kernel_0)(0,
        napalm::ImgRegion( m_height / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));
    
    (*m_compute_stereo_horizontal_dir_kernel_4)(0,
        napalm::ImgRegion(m_height / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));

    (*m_compute_stereo_vertical_dir_kernel_2)(0,
        napalm::ImgRegion(m_width / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));

    (*m_compute_stereo_vertical_dir_kernel_6)(0,
        napalm::ImgRegion(m_width / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));

    const int obl_num_paths = m_width + m_height ;
    (*m_compute_stereo_oblique_dir_kernel_1)(0,
        napalm::ImgRegion(obl_num_paths / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));

    (*m_compute_stereo_oblique_dir_kernel_3)(0,
        napalm::ImgRegion(obl_num_paths / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));


    (*m_compute_stereo_oblique_dir_kernel_5)(0,
        napalm::ImgRegion(obl_num_paths / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));

    (*m_compute_stereo_oblique_dir_kernel_7)(0,
        napalm::ImgRegion( obl_num_paths / PATHS_IN_BLOCK, 1),
        napalm::ImgRegion(32, PATHS_IN_BLOCK));

}

void StereoSGMCL::winner_takes_all()
{
    const int WTA_PIXEL_IN_BLOCK = 8;
    (*m_winner_takes_all_kernel128)(0,
        napalm::ImgRegion( m_width / WTA_PIXEL_IN_BLOCK, 1 * m_height),
        napalm::ImgRegion(32, WTA_PIXEL_IN_BLOCK));
}

void StereoSGMCL::median()
{
    m_median_3x3->setArgs(d_left_disparity, d_tmp_left_disp);
    (*m_median_3x3)(0,
        napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
        napalm::ImgRegion(16, 16));

    m_median_3x3->setArgs(d_right_disparity, d_tmp_right_disp);
    (*m_median_3x3)(0,
        napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
        napalm::ImgRegion(16, 16));

}

void StereoSGMCL::check_consistency_left()
{
    (*m_check_consistency_left)(0,
        napalm::ImgRegion((m_width + 16 - 1) / 16, (m_height + 16 - 1) / 16),
        napalm::ImgRegion(16, 16));
}
