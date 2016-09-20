#include "stereo-sgm.h"
#include <vector>
#include <assert.h>
#include <fstream>

std::string cl_file_name = "sgm-program.cl";
int platform_index = 0;
int device_index = 0;

const char * cl_error_strings_helper[] =
{
	"CL_SUCCESS",
	"CL_DEVICE_NOT_FOUND",
	"CL_DEVICE_NOT_AVAILABLE",
	"CL_COMPILER_NOT_AVAILABLE",
	"CL_MEM_OBJECT_ALLOCATION_FAILURE",
	"CL_OUT_OF_RESOURCES",
	"CL_OUT_OF_HOST_MEMORY",
	"CL_PROFILING_INFO_NOT_AVAILABLE",
	"CL_MEM_COPY_OVERLAP",
	"CL_IMAGE_FORMAT_MISMATCH",
	"CL_IMAGE_FORMAT_NOT_SUPPORTED",
	"CL_BUILD_PROGRAM_FAILURE",
	"CL_MAP_FAILURE",
	"CL_MISALIGNED_SUB_BUFFER_OFFSET",
	"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	" ",
	"CL_INVALID_VALUE",
	"CL_INVALID_DEVICE_TYPE",
	"CL_INVALID_PLATFORM",
	"CL_INVALID_DEVICE",
	"CL_INVALID_CONTEXT",
	"CL_INVALID_QUEUE_PROPERTIES",
	"CL_INVALID_COMMAND_QUEUE",
	"CL_INVALID_HOST_PTR",
	"CL_INVALID_MEM_OBJECT",
	"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
	"CL_INVALID_IMAGE_SIZE",
	"CL_INVALID_SAMPLER",
	"CL_INVALID_BINARY",
	"CL_INVALID_BUILD_OPTIONS",
	"CL_INVALID_PROGRAM",
	"CL_INVALID_PROGRAM_EXECUTABLE",
	"CL_INVALID_KERNEL_NAME",
	"CL_INVALID_KERNEL_DEFINITION",
	"CL_INVALID_KERNEL",
	"CL_INVALID_ARG_INDEX",
	"CL_INVALID_ARG_VALUE",
	"CL_INVALID_ARG_SIZE",
	"CL_INVALID_KERNEL_ARGS",
	"CL_INVALID_WORK_DIMENSION",
	"CL_INVALID_WORK_GROUP_SIZE",
	"CL_INVALID_WORK_ITEM_SIZE",
	"CL_INVALID_GLOBAL_OFFSET",
	"CL_INVALID_EVENT_WAIT_LIST",
	"CL_INVALID_EVENT",
	"CL_INVALID_OPERATION",
	"CL_INVALID_GL_OBJECT",
	"CL_INVALID_BUFFER_SIZE",
	"CL_INVALID_MIP_LEVEL",
	"CL_INVALID_GLOBAL_WORK_SIZE",
	"CL_INVALID_PROPERTY"
};

const std::vector<std::string> cl_error_strings(cl_error_strings_helper, std::end(cl_error_strings_helper));

void checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS)
	{
		printf("CL error: %s , error %d, %s ", name, err, cl_error_strings[-1 * err].c_str());
	}
}


StereoSGM::StereoSGM(int width, int height, int max_disp_size):
	m_width(width), m_height(height), m_max_disparity(max_disp_size)
{
	initCL();
}


StereoSGM::~StereoSGM()
{
}

bool StereoSGM::init()
{
	initCL();
	return true;
}

void StereoSGM::execute(void * left_data, void * right_data, void * output_buffer)
{
	cl_int m_err = m_command_queue.finish();
	m_err = m_command_queue.enqueueWriteBuffer(d_src_left, true, 0, m_width * m_height, left_data);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	m_err = m_command_queue.enqueueWriteBuffer(d_src_right, true, 0, m_width * m_height, right_data);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	m_err = m_command_queue.finish();
	census();
	m_err = m_command_queue.finish();


	mem_init();
	m_err = m_command_queue.finish();

	matching_cost();
	m_err = m_command_queue.finish();

	
	//m_err = m_command_queue.enqueueNDRangeKernel(m_copy_u8_to_u16,
	//	cl::NDRange(0, 0, 0),
	//	cl::NDRange(128 * m_width * m_height),
	//	cl::NDRange(128));
	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());


	scan_cost();
	m_err = m_command_queue.finish();

	winner_takes_all();
	m_err = m_command_queue.finish();

	median();
	m_err = m_command_queue.finish();

	check_consistency_left();

	m_err = m_command_queue.finish();
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());


	m_command_queue.enqueueReadBuffer(d_tmp_left_disp, true, 0, sizeof(uint16_t) * m_width * m_height, output_buffer);

}

void StereoSGM::initCL()
{
	std::vector<cl::Platform> platform_list;
	cl::Platform::get(&platform_list);

	assert(platform_index < platform_list.size());

	std::string platformVendor;
	std::string platformVersion;

	platform_list[platform_index].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
	printf("Platform vendor: %s \n", platformVendor.c_str());

	platform_list[platform_index].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVersion);
	printf("Platform version: %s \n", platformVersion.c_str());

	std::vector<cl::Device> devices;
	cl_int m_err = platform_list[platform_index].getDevices(
		CL_DEVICE_TYPE_ALL, &devices);
	checkErr(m_err, "getDevicesList");

	assert(devices.size() > device_index);
	m_device = devices[device_index];
	std::string device_name;
	m_device.getInfo(CL_DEVICE_NAME, &device_name);
	printf("Selected device: %s \n", device_name.c_str());


	cl_context_properties cprops[3] =
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_list[platform_index])(), 0
	};

	m_context = cl::Context(
		m_device,
		cprops,
		NULL,
		NULL,
		&m_err
	);

	checkErr(m_err, "Conext::Context()");

	m_command_queue = cl::CommandQueue(m_context, m_device, 0, &m_err);
	checkErr(m_err, "COMMAND QUEUE creation failed");
	size_t max_wg_size = 0;
	m_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_wg_size);
	printf("OpenCL context created, max wg size: %zd \n", max_wg_size);


	//reading the kernel source
	std::ifstream f(cl_file_name);
	std::string kernel_source((std::istreambuf_iterator<char>(f)),
		std::istreambuf_iterator<char>());

	cl::Program::Sources source(1,
		std::make_pair(kernel_source.c_str(), kernel_source.size()));
	m_program = cl::Program(m_context, source);
	m_err = m_program.build({ m_device });
	checkErr(m_err, "build kernel");

	std::string build_log = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
	if (!build_log.empty() && build_log != "\n")
		printf("CL kernel build LOG: \n %s \n", build_log.c_str());

	m_census_kernel = cl::Kernel(m_program, "census_kernel", &m_err);
	checkErr(m_err, "kernel");
	m_matching_cost_kernel_128 = cl::Kernel(m_program, "matching_cost_kernel_128", &m_err);
	checkErr(m_err, "kernel");
	
	
	m_compute_stereo_horizontal_dir_kernel_0 = cl::Kernel(m_program, "compute_stereo_horizontal_dir_kernel_0", &m_err);
	checkErr(m_err, "kernel");
	m_compute_stereo_horizontal_dir_kernel_4 = cl::Kernel(m_program, "compute_stereo_horizontal_dir_kernel_4", &m_err);
	checkErr(m_err, "kernel");
	m_compute_stereo_vertical_dir_kernel_2 = cl::Kernel(m_program, "compute_stereo_vertical_dir_kernel_2", &m_err);
	checkErr(m_err, "kernel");
	m_compute_stereo_vertical_dir_kernel_6 = cl::Kernel(m_program, "compute_stereo_vertical_dir_kernel_6", &m_err);
	checkErr(m_err, "kernel");	
	m_compute_stereo_oblique_dir_kernel_1 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_1", &m_err);
	checkErr(m_err, "kernel");
	m_compute_stereo_oblique_dir_kernel_3 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_3", &m_err);
	checkErr(m_err, "kernel");
	m_compute_stereo_oblique_dir_kernel_5 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_5", &m_err);
	checkErr(m_err, "kernel");
	m_compute_stereo_oblique_dir_kernel_7 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_7", &m_err);
	checkErr(m_err, "kernel");


	m_winner_takes_all_kernel128 = cl::Kernel(m_program, "winner_takes_all_kernel128", &m_err);
	checkErr(m_err, "kernel");

	m_check_consistency_left = cl::Kernel(m_program, "check_consistency_kernel_left", &m_err);
	checkErr(m_err, "kernel");

	m_median_3x3 = cl::Kernel(m_program, "median3x3", &m_err);
	checkErr(m_err, "kernel");

	m_copy_u8_to_u16 = cl::Kernel(m_program, "copy_u8_to_u16", &m_err);
	checkErr(m_err, "kernel");

	d_src_left = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_width * m_height);
	d_src_right = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_width * m_height);

	d_left = cl::Buffer(m_context, CL_MEM_READ_WRITE, sizeof(uint64_t) * m_width * m_height);
	d_right = cl::Buffer(m_context, CL_MEM_READ_WRITE, sizeof(uint64_t) * m_width * m_height);

	d_matching_cost = cl::Buffer(m_context, CL_MEM_READ_WRITE, 
		m_width * m_height * m_max_disparity);

	d_scost = cl::Buffer(m_context, CL_MEM_READ_WRITE, 
		sizeof(uint16_t) * m_width * m_height * m_max_disparity);

	d_left_disparity = cl::Buffer(m_context, CL_MEM_READ_WRITE,
		sizeof(uint16_t) * m_width * m_height);

	d_right_disparity = cl::Buffer(m_context, CL_MEM_READ_WRITE,
		sizeof(uint16_t) * m_width * m_height);

	d_tmp_left_disp = cl::Buffer(m_context, CL_MEM_READ_WRITE,
		sizeof(uint16_t) * m_width * m_height);

	d_tmp_right_disp = cl::Buffer(m_context, CL_MEM_READ_WRITE,
		sizeof(uint16_t) * m_width * m_height);

	//setup kernels
	m_census_kernel.setArg(0, d_src_left);
	m_census_kernel.setArg(1, d_left);
	m_census_kernel.setArg(2, m_width);
	m_census_kernel.setArg(3, m_height);

	m_matching_cost_kernel_128.setArg(0, d_left);
	m_matching_cost_kernel_128.setArg(1, d_right);
	m_matching_cost_kernel_128.setArg(2, d_matching_cost);
	m_matching_cost_kernel_128.setArg(3, m_width);
	m_matching_cost_kernel_128.setArg(4, m_height);


	m_compute_stereo_horizontal_dir_kernel_0.setArg(0, d_matching_cost);
	m_compute_stereo_horizontal_dir_kernel_0.setArg(1, d_scost);
	m_compute_stereo_horizontal_dir_kernel_0.setArg(2, m_width);
	m_compute_stereo_horizontal_dir_kernel_0.setArg(3, m_height);

	m_compute_stereo_horizontal_dir_kernel_4.setArg(0, d_matching_cost);
	m_compute_stereo_horizontal_dir_kernel_4.setArg(1, d_scost);
	m_compute_stereo_horizontal_dir_kernel_4.setArg(2, m_width);
	m_compute_stereo_horizontal_dir_kernel_4.setArg(3, m_height);


	m_compute_stereo_vertical_dir_kernel_2.setArg(0, d_matching_cost);
	m_compute_stereo_vertical_dir_kernel_2.setArg(1, d_scost);
	m_compute_stereo_vertical_dir_kernel_2.setArg(2, m_width);
	m_compute_stereo_vertical_dir_kernel_2.setArg(3, m_height);

	m_compute_stereo_vertical_dir_kernel_6.setArg(0, d_matching_cost);
	m_compute_stereo_vertical_dir_kernel_6.setArg(1, d_scost);
	m_compute_stereo_vertical_dir_kernel_6.setArg(2, m_width);
	m_compute_stereo_vertical_dir_kernel_6.setArg(3, m_height);


	m_compute_stereo_oblique_dir_kernel_1.setArg(0, d_matching_cost);
	m_compute_stereo_oblique_dir_kernel_1.setArg(1, d_scost);
	m_compute_stereo_oblique_dir_kernel_1.setArg(2, m_width);
	m_compute_stereo_oblique_dir_kernel_1.setArg(3, m_height);

	m_compute_stereo_oblique_dir_kernel_3.setArg(0, d_matching_cost);
	m_compute_stereo_oblique_dir_kernel_3.setArg(1, d_scost);
	m_compute_stereo_oblique_dir_kernel_3.setArg(2, m_width);
	m_compute_stereo_oblique_dir_kernel_3.setArg(3, m_height);

	m_compute_stereo_oblique_dir_kernel_5.setArg(0, d_matching_cost);
	m_compute_stereo_oblique_dir_kernel_5.setArg(1, d_scost);
	m_compute_stereo_oblique_dir_kernel_5.setArg(2, m_width);
	m_compute_stereo_oblique_dir_kernel_5.setArg(3, m_height);

	m_compute_stereo_oblique_dir_kernel_7.setArg(0, d_matching_cost);
	m_compute_stereo_oblique_dir_kernel_7.setArg(1, d_scost);
	m_compute_stereo_oblique_dir_kernel_7.setArg(2, m_width);
	m_compute_stereo_oblique_dir_kernel_7.setArg(3, m_height);



	m_winner_takes_all_kernel128.setArg(0, d_left_disparity);
	m_winner_takes_all_kernel128.setArg(1, d_right_disparity);
	m_winner_takes_all_kernel128.setArg(2, d_scost);
	m_winner_takes_all_kernel128.setArg(3, m_width);
	m_winner_takes_all_kernel128.setArg(4, m_height);


	m_check_consistency_left.setArg(0, d_tmp_left_disp);
	m_check_consistency_left.setArg(1, d_tmp_right_disp);
	m_check_consistency_left.setArg(2, d_src_left);
	m_check_consistency_left.setArg(3, m_width);
	m_check_consistency_left.setArg(4, m_height);
	
	m_median_3x3.setArg(0, d_left_disparity);
	m_median_3x3.setArg(1, d_tmp_left_disp);
	m_median_3x3.setArg(2, m_width);
	m_median_3x3.setArg(3, m_height);


	m_copy_u8_to_u16.setArg(0, d_matching_cost);
	m_copy_u8_to_u16.setArg(1, d_scost);
}

void StereoSGM::census()
{
	//setup kernels
	m_census_kernel.setArg(0, d_src_left);
	m_census_kernel.setArg(1, d_left);
	cl::Event census_event;
	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_census_kernel, cl::NDRange(0, 0, 0),
		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
		cl::NDRange(16, 16), nullptr, &census_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	census_event;
	m_census_kernel.setArg(0, d_src_right);
	m_census_kernel.setArg(1, d_right);
	m_err = m_command_queue.enqueueNDRangeKernel(m_census_kernel, cl::NDRange(0, 0, 0),
		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
		cl::NDRange(16, 16), nullptr, &census_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGM::mem_init()
{
	cl_int m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_left_disparity, 0, 0,
		m_width * m_height * sizeof(uint16_t));
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
	m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_right_disparity, 0, 0,
		m_width * m_height * sizeof(uint16_t));
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_scost, 0, 0,
		m_width * m_height * sizeof(uint16_t) * m_max_disparity);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGM::matching_cost()
{
	int MCOST_LINES128 = 8;
	cl::Event matching_cost_event;
	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_matching_cost_kernel_128, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * m_height / MCOST_LINES128, MCOST_LINES128),
		cl::NDRange(32, MCOST_LINES128), nullptr, &matching_cost_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGM::scan_cost()
{
	//census_event.wait();
	static const int PATHS_IN_BLOCK = 8;
	cl::Event scan_cost_0_event;
	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_horizontal_dir_kernel_0, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * m_height / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_0_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
	
	
	cl::Event scan_cost_4_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_horizontal_dir_kernel_4, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * m_height / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_4_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
	
	cl::Event scan_cost_2_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_vertical_dir_kernel_2, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * m_width / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_2_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
	
	cl::Event scan_cost_6_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_vertical_dir_kernel_6, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * m_width / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_6_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	const int obl_num_paths = m_width + m_height ;
	cl::Event scan_cost_1_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_1, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_1_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	cl::Event scan_cost_3_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_3, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_3_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());


	cl::Event scan_cost_5_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_5, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_5_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

	cl::Event scan_cost_7_event;
	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_7, cl::NDRange(0, 0, 0),
		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_7_event);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());



}

void StereoSGM::winner_takes_all()
{
	const int WTA_PIXEL_IN_BLOCK = 8;
	cl::Event winner_takes_it_all;
	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_winner_takes_all_kernel128,
		cl::NDRange(0, 0, 0),
		cl::NDRange(32 * m_width / WTA_PIXEL_IN_BLOCK, WTA_PIXEL_IN_BLOCK * m_height),
		cl::NDRange(32, WTA_PIXEL_IN_BLOCK), nullptr, &winner_takes_it_all);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGM::median()
{
	m_median_3x3.setArg(0, d_left_disparity);
	m_median_3x3.setArg(1, d_tmp_left_disp);
	cl::Event median_ev;
	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_median_3x3,
		cl::NDRange(0, 0, 0),
		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
		cl::NDRange(16, 16), nullptr, &median_ev);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());



	m_median_3x3.setArg(0, d_right_disparity);
	m_median_3x3.setArg(1, d_tmp_right_disp);
	m_err = m_command_queue.enqueueNDRangeKernel(m_median_3x3,
		cl::NDRange(0, 0, 0),
		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
		cl::NDRange(16, 16), nullptr, &median_ev);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());

}

void StereoSGM::check_consistency_left()
{
	cl::Event check_consistency_kernel_left_ev;
	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_check_consistency_left,
		cl::NDRange(0, 0, 0),
		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
		cl::NDRange(16, 16), nullptr, &check_consistency_kernel_left_ev);
	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}
