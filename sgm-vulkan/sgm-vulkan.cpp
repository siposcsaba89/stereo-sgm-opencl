#include "sgm-vulkan.h"
#include <vector>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#ifdef _MSC_VER
#include <Windows.h>
#endif
//int platform_index = 1;
int device_index = 1;
//
//void checkErr(cl_int err, const char * name)
//{
//	if (err != CL_SUCCESS)
//	{
//		printf("CL error: %s , error %d, %s ", name, err, cl_error_strings[-1 * err].c_str());
//	}
//}
//

VkBool32 messageCallback(
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT objType,
    uint64_t srcObject,
    size_t location,
    int32_t msgCode,
    const char* pLayerPrefix,
    const char* pMsg,
    void* pUserData) {
    std::string message;
    {
        std::stringstream buf;
        if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
            buf << "ERROR: ";
        }
        else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
            buf << "WARNING: ";
        }
        else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
            buf << "PERF: ";
        }
        else {
            return false;
        }
        buf << "[" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;
        message = buf.str();
    }

    std::cout << message << std::endl;
#ifdef _MSC_VER 
    OutputDebugStringA(message.c_str());
    OutputDebugStringA("\n");
#endif
    return false;
}

StereoSGMVULKAN::StereoSGMVULKAN(int width, int height, int max_disp_size):
	m_width(width), m_height(height), m_max_disparity(max_disp_size)
{
	init();
}


StereoSGMVULKAN::~StereoSGMVULKAN()
{
}

bool StereoSGMVULKAN::init()
{
	initVulkan();
	return true;
}

void StereoSGMVULKAN::execute(void * left_data, void * right_data, void * output_buffer)
{
//	cl_int m_err = m_command_queue.finish();
//	m_err = m_command_queue.enqueueWriteBuffer(d_src_left, true, 0, m_width * m_height, left_data);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//	m_err = m_command_queue.enqueueWriteBuffer(d_src_right, true, 0, m_width * m_height, right_data);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//	m_err = m_command_queue.finish();
//	census();
//	m_err = m_command_queue.finish();
//
//
//	mem_init();
//	m_err = m_command_queue.finish();
//
//	matching_cost();
//	m_err = m_command_queue.finish();
//
//	
//	//m_err = m_command_queue.enqueueNDRangeKernel(m_copy_u8_to_u16,
//	//	cl::NDRange(0, 0, 0),
//	//	cl::NDRange(128 * m_width * m_height),
//	//	cl::NDRange(128));
//	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//
//	scan_cost();
//	m_err = m_command_queue.finish();
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	winner_takes_all();
//	m_err = m_command_queue.finish();
//
//	median();
//	m_err = m_command_queue.finish();
//
//	check_consistency_left();
//
//	m_err = m_command_queue.finish();
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//
//	m_command_queue.enqueueReadBuffer(d_tmp_left_disp, true, 0, sizeof(uint16_t) * m_width * m_height, output_buffer);
//
}

void StereoSGMVULKAN::initVulkan()
{

    { // instacnce creation

        m_enable_validation = true;
        std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

        printf("Available Vulkan instance extensions: \n");

        for (auto ext : extensions)
        {
            printf("\t - %s \n", ext.extensionName);
        }

        std::vector<vk::LayerProperties> layers = vk::enumerateInstanceLayerProperties();

        printf("Available Vulkan instance layers: \n");

        for (auto ext : layers)
        {
            printf("\t - %s \n", ext.layerName);
        }
        std::vector<const char*> enabledExtensions;
        if (m_enable_validation)
        {
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }

        vk::InstanceCreateInfo info(
            vk::InstanceCreateFlags(),
            &vk::ApplicationInfo("sgm-vulkan", VK_MAKE_VERSION(0, 1, 0), "sgm-vulkan-engine", VK_MAKE_VERSION(0, 1, 0), VK_API_VERSION_1_0),
            0,
            nullptr,
            uint32_t(enabledExtensions.size()),
            enabledExtensions.data()
        );

        m_vk_instance = vk::createInstance(info);
    }

    {//device creation
        std::vector<vk::PhysicalDevice> physical_devices = m_vk_instance.enumeratePhysicalDevices();
        m_vk_physical_device = physical_devices[device_index];

        vk::PhysicalDeviceProperties prop = m_vk_physical_device.getProperties();

        printf("Selected device: %s \n", prop.deviceName);

        vk::PhysicalDeviceProperties dev_props = m_vk_physical_device.getProperties();

        
        printf("Api version: %d %d %d \n", (dev_props.apiVersion >> 22) & 0x7FF, (dev_props.apiVersion >> 12) & 0xFF, dev_props.apiVersion & 0xFFF);

        std::vector<vk::ExtensionProperties> dev_exts = m_vk_physical_device.enumerateDeviceExtensionProperties();
        printf("Selected device extensions: \n");
        for (auto ext : dev_exts)
        {
            printf("\t - %s \n", ext.extensionName);
        }

        std::vector<vk::LayerProperties> dev_layers = m_vk_physical_device.enumerateDeviceLayerProperties();
        printf("Selected device layers: \n");
        for (auto lay : dev_layers)
        {
            printf("\t - %s \n", lay.layerName);
        }

        
        int compute_queue_idx = 0;

        std::vector<vk::QueueFamilyProperties> queueProps = m_vk_physical_device.getQueueFamilyProperties();
        size_t queueCount = queueProps.size();
        for (uint32_t i = 0; i < queueCount; ++i) {
            if (queueProps[i].queueFlags & vk::QueueFlagBits::eCompute) {
                compute_queue_idx = i;
                break;
            }
        }
        std::array<float, 1> queuePriorities = { 0.0f };
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.queueFamilyIndex = compute_queue_idx;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = queuePriorities.data();
//later check if extensions are available or not
        std::vector<const char*> enabledExtensions = { 
            "VK_AMD_shader_ballot", 
            "VK_AMD_shader_trinary_minmax", 
            "VK_AMD_gcn_shader",
            "VK_AMD_gpu_shader_half_float"
        };
        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        m_vk_device_features = m_vk_physical_device.getFeatures();
        deviceCreateInfo.pEnabledFeatures = &m_vk_device_features;
        // enable the debug marker extension if it is present (likely meaning a debugging tool is present)
        if (enabledExtensions.size() > 0) {
            deviceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
            deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
        }
        if (m_enable_validation)
        {
            std::vector<const char*> validationLayerNames = { {
                    // This is a meta layer that enables all of the standard
                    // validation layers in the correct order :
                    // threading, parameter_validation, device_limits, object_tracker, image, core_validation, swapchain, and unique_objects
                    "VK_LAYER_LUNARG_standard_validation"
                } };
            deviceCreateInfo.enabledLayerCount = uint32_t(validationLayerNames.size());
            deviceCreateInfo.ppEnabledLayerNames = validationLayerNames.data();
        }

        m_vk_device = m_vk_physical_device.createDevice(deviceCreateInfo);
        if (m_enable_validation)
        {
            CreateDebugReportCallback = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(m_vk_instance, "vkCreateDebugReportCallbackEXT");
            DestroyDebugReportCallback = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(m_vk_instance, "vkDestroyDebugReportCallbackEXT");
            dbgBreakCallback = (PFN_vkDebugReportMessageEXT)vkGetInstanceProcAddr(m_vk_instance, "vkDebugReportMessageEXT");

            VkDebugReportCallbackCreateInfoEXT dbgCreateInfo = {};
            dbgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
            dbgCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)messageCallback;
            vk::DebugReportFlagsEXT flags = vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning;
            dbgCreateInfo.flags = flags.operator VkSubpassDescriptionFlags();

            VkResult err = CreateDebugReportCallback(
                m_vk_instance,
                &dbgCreateInfo,
                nullptr,
                &msgCallback);
            assert(!err);
        }
        m_vk_queue = m_vk_device.getQueue(compute_queue_idx, 0);
        vk::SubmitInfo info;
        vk::CommandBuffer cmdb;
        vk::CommandBufferBeginInfo beg_info;
        cmdb.begin(beg_info);
        cmdb.end();
        vk::Fence f;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &cmdb;
        m_vk_queue.submit(1, &info, f);

        m_vk_queue.waitIdle();

    }


//	std::vector<cl::Platform> platform_list;
//	cl::Platform::get(&platform_list);
//
//	assert(platform_index < platform_list.size());
//
//	std::string platformVendor;
//	std::string platformVersion;
//
//	platform_list[platform_index].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
//	printf("Platform vendor: %s \n", platformVendor.c_str());
//
//	platform_list[platform_index].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVersion);
//	printf("Platform version: %s \n", platformVersion.c_str());
//
//	std::vector<cl::Device> devices;
//	cl_int m_err = platform_list[platform_index].getDevices(
//		CL_DEVICE_TYPE_ALL, &devices);
//	checkErr(m_err, "getDevicesList");
//
//	assert(devices.size() > device_index);
//	m_device = devices[device_index];
//	std::string device_name;
//	m_device.getInfo(CL_DEVICE_NAME, &device_name);
//	printf("Selected device: %s \n", device_name.c_str());
//
//
//	cl_context_properties cprops[3] =
//	{
//		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_list[platform_index])(), 0
//	};
//
//	m_context = cl::Context(
//		m_device,
//		cprops,
//		NULL,
//		NULL,
//		&m_err
//	);
//
//	checkErr(m_err, "Conext::Context()");
//
//	m_command_queue = cl::CommandQueue(m_context, m_device, 0, &m_err);
//	checkErr(m_err, "COMMAND QUEUE creation failed");
//	size_t max_wg_size = 0;
//	m_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_wg_size);
//	printf("OpenCL context created, max wg size: %zd \n", max_wg_size);
//
//
//	//reading the kernel source
//	std::ifstream f(cl_file_name);
//	std::string kernel_source((std::istreambuf_iterator<char>(f)),
//		std::istreambuf_iterator<char>());
//
//	cl::Program::Sources source(1,
//		std::make_pair(kernel_source.c_str(), kernel_source.size()));
//	m_program = cl::Program(m_context, source);
//	m_err = m_program.build({ m_device });
//	checkErr(m_err, "build kernel");
//
//	std::string build_log = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
//	if (!build_log.empty() && build_log != "\n")
//		printf("CL kernel build LOG: \n %s \n", build_log.c_str());
//
//	m_census_kernel = cl::Kernel(m_program, "census_kernel", &m_err);
//	checkErr(m_err, "kernel");
//	m_matching_cost_kernel_128 = cl::Kernel(m_program, "matching_cost_kernel_128", &m_err);
//	checkErr(m_err, "kernel");
//	
//	
//	m_compute_stereo_horizontal_dir_kernel_0 = cl::Kernel(m_program, "compute_stereo_horizontal_dir_kernel_0", &m_err);
//	checkErr(m_err, "kernel");
//	m_compute_stereo_horizontal_dir_kernel_4 = cl::Kernel(m_program, "compute_stereo_horizontal_dir_kernel_4", &m_err);
//	checkErr(m_err, "kernel");
//	m_compute_stereo_vertical_dir_kernel_2 = cl::Kernel(m_program, "compute_stereo_vertical_dir_kernel_2", &m_err);
//	checkErr(m_err, "kernel");
//	m_compute_stereo_vertical_dir_kernel_6 = cl::Kernel(m_program, "compute_stereo_vertical_dir_kernel_6", &m_err);
//	checkErr(m_err, "kernel");	
//	m_compute_stereo_oblique_dir_kernel_1 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_1", &m_err);
//	checkErr(m_err, "kernel");
//	m_compute_stereo_oblique_dir_kernel_3 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_3", &m_err);
//	checkErr(m_err, "kernel");
//	m_compute_stereo_oblique_dir_kernel_5 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_5", &m_err);
//	checkErr(m_err, "kernel");
//	m_compute_stereo_oblique_dir_kernel_7 = cl::Kernel(m_program, "compute_stereo_oblique_dir_kernel_7", &m_err);
//	checkErr(m_err, "kernel");
//
//
//	m_winner_takes_all_kernel128 = cl::Kernel(m_program, "winner_takes_all_kernel128", &m_err);
//	checkErr(m_err, "kernel");
//
//	m_check_consistency_left = cl::Kernel(m_program, "check_consistency_kernel_left", &m_err);
//	checkErr(m_err, "kernel");
//
//	m_median_3x3 = cl::Kernel(m_program, "median3x3", &m_err);
//	checkErr(m_err, "kernel");
//
//	m_copy_u8_to_u16 = cl::Kernel(m_program, "copy_u8_to_u16", &m_err);
//	checkErr(m_err, "kernel");
//
//	m_clear_buffer = cl::Kernel(m_program, "clear_buffer", &m_err);
//	checkErr(m_err, "kernel");
//
//
//	d_src_left = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_width * m_height);
//	d_src_right = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_width * m_height);
//
//	d_left = cl::Buffer(m_context, CL_MEM_READ_WRITE, sizeof(uint64_t) * m_width * m_height);
//	d_right = cl::Buffer(m_context, CL_MEM_READ_WRITE, sizeof(uint64_t) * m_width * m_height);
//
//	d_matching_cost = cl::Buffer(m_context, CL_MEM_READ_WRITE, 
//		m_width * m_height * m_max_disparity);
//
//	d_scost = cl::Buffer(m_context, CL_MEM_READ_WRITE, 
//		sizeof(uint16_t) * m_width * m_height * m_max_disparity);
//
//	d_left_disparity = cl::Buffer(m_context, CL_MEM_READ_WRITE,
//		sizeof(uint16_t) * m_width * m_height);
//
//	d_right_disparity = cl::Buffer(m_context, CL_MEM_READ_WRITE,
//		sizeof(uint16_t) * m_width * m_height);
//
//	d_tmp_left_disp = cl::Buffer(m_context, CL_MEM_READ_WRITE,
//		sizeof(uint16_t) * m_width * m_height);
//
//	d_tmp_right_disp = cl::Buffer(m_context, CL_MEM_READ_WRITE,
//		sizeof(uint16_t) * m_width * m_height);
//
//	//setup kernels
//	m_census_kernel.setArg(0, d_src_left);
//	m_census_kernel.setArg(1, d_left);
//	m_census_kernel.setArg(2, m_width);
//	m_census_kernel.setArg(3, m_height);
//
//	m_matching_cost_kernel_128.setArg(0, d_left);
//	m_matching_cost_kernel_128.setArg(1, d_right);
//	m_matching_cost_kernel_128.setArg(2, d_matching_cost);
//	m_matching_cost_kernel_128.setArg(3, m_width);
//	m_matching_cost_kernel_128.setArg(4, m_height);
//
//
//	m_compute_stereo_horizontal_dir_kernel_0.setArg(0, d_matching_cost);
//	m_compute_stereo_horizontal_dir_kernel_0.setArg(1, d_scost);
//	m_compute_stereo_horizontal_dir_kernel_0.setArg(2, m_width);
//	m_compute_stereo_horizontal_dir_kernel_0.setArg(3, m_height);
//
//	m_compute_stereo_horizontal_dir_kernel_4.setArg(0, d_matching_cost);
//	m_compute_stereo_horizontal_dir_kernel_4.setArg(1, d_scost);
//	m_compute_stereo_horizontal_dir_kernel_4.setArg(2, m_width);
//	m_compute_stereo_horizontal_dir_kernel_4.setArg(3, m_height);
//
//
//	m_compute_stereo_vertical_dir_kernel_2.setArg(0, d_matching_cost);
//	m_compute_stereo_vertical_dir_kernel_2.setArg(1, d_scost);
//	m_compute_stereo_vertical_dir_kernel_2.setArg(2, m_width);
//	m_compute_stereo_vertical_dir_kernel_2.setArg(3, m_height);
//
//	m_compute_stereo_vertical_dir_kernel_6.setArg(0, d_matching_cost);
//	m_compute_stereo_vertical_dir_kernel_6.setArg(1, d_scost);
//	m_compute_stereo_vertical_dir_kernel_6.setArg(2, m_width);
//	m_compute_stereo_vertical_dir_kernel_6.setArg(3, m_height);
//
//
//	m_compute_stereo_oblique_dir_kernel_1.setArg(0, d_matching_cost);
//	m_compute_stereo_oblique_dir_kernel_1.setArg(1, d_scost);
//	m_compute_stereo_oblique_dir_kernel_1.setArg(2, m_width);
//	m_compute_stereo_oblique_dir_kernel_1.setArg(3, m_height);
//
//	m_compute_stereo_oblique_dir_kernel_3.setArg(0, d_matching_cost);
//	m_compute_stereo_oblique_dir_kernel_3.setArg(1, d_scost);
//	m_compute_stereo_oblique_dir_kernel_3.setArg(2, m_width);
//	m_compute_stereo_oblique_dir_kernel_3.setArg(3, m_height);
//
//	m_compute_stereo_oblique_dir_kernel_5.setArg(0, d_matching_cost);
//	m_compute_stereo_oblique_dir_kernel_5.setArg(1, d_scost);
//	m_compute_stereo_oblique_dir_kernel_5.setArg(2, m_width);
//	m_compute_stereo_oblique_dir_kernel_5.setArg(3, m_height);
//
//	m_compute_stereo_oblique_dir_kernel_7.setArg(0, d_matching_cost);
//	m_compute_stereo_oblique_dir_kernel_7.setArg(1, d_scost);
//	m_compute_stereo_oblique_dir_kernel_7.setArg(2, m_width);
//	m_compute_stereo_oblique_dir_kernel_7.setArg(3, m_height);
//
//
//
//	m_winner_takes_all_kernel128.setArg(0, d_left_disparity);
//	m_winner_takes_all_kernel128.setArg(1, d_right_disparity);
//	m_winner_takes_all_kernel128.setArg(2, d_scost);
//	m_winner_takes_all_kernel128.setArg(3, m_width);
//	m_winner_takes_all_kernel128.setArg(4, m_height);
//
//
//	m_check_consistency_left.setArg(0, d_tmp_left_disp);
//	m_check_consistency_left.setArg(1, d_tmp_right_disp);
//	m_check_consistency_left.setArg(2, d_src_left);
//	m_check_consistency_left.setArg(3, m_width);
//	m_check_consistency_left.setArg(4, m_height);
//	
//	m_median_3x3.setArg(0, d_left_disparity);
//	m_median_3x3.setArg(1, d_tmp_left_disp);
//	m_median_3x3.setArg(2, m_width);
//	m_median_3x3.setArg(3, m_height);
//
//
//	m_copy_u8_to_u16.setArg(0, d_matching_cost);
//	m_copy_u8_to_u16.setArg(1, d_scost);
}

void StereoSGMVULKAN::census()
{
//	//setup kernels
//	m_census_kernel.setArg(0, d_src_left);
//	m_census_kernel.setArg(1, d_left);
//	cl::Event census_event;
//	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_census_kernel, cl::NDRange(0, 0, 0),
//		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
//		cl::NDRange(16, 16), nullptr, &census_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//	census_event;
//	m_census_kernel.setArg(0, d_src_right);
//	m_census_kernel.setArg(1, d_right);
//	m_err = m_command_queue.enqueueNDRangeKernel(m_census_kernel, cl::NDRange(0, 0, 0),
//		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
//		cl::NDRange(16, 16), nullptr, &census_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGMVULKAN::mem_init()
{
//	//cl_int m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_left_disparity, 0, 0,
//	//	m_width * m_height * sizeof(uint16_t));
//	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	//m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_right_disparity, 0, 0,
//	//	m_width * m_height * sizeof(uint16_t));
//	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	//
//	//m_err = m_command_queue.enqueueFillBuffer<cl_uchar>(d_scost, 0, 0,
//	//	m_width * m_height * sizeof(uint16_t) * m_max_disparity);
//	//checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//	m_clear_buffer.setArg(0, d_left_disparity);
//	m_command_queue.enqueueNDRangeKernel(m_clear_buffer, cl::NDRange(0, 0, 0),
//		cl::NDRange(m_width * m_height * sizeof(uint16_t) / 32),
//		cl::NDRange(256));
//	m_clear_buffer.setArg(0, d_right_disparity);
//	m_command_queue.enqueueNDRangeKernel(m_clear_buffer, cl::NDRange(0, 0, 0),
//		cl::NDRange(m_width * m_height * sizeof(uint16_t) / 32),
//		cl::NDRange(256));
//	
//	m_clear_buffer.setArg(0, d_scost);
//	m_command_queue.enqueueNDRangeKernel(m_clear_buffer, cl::NDRange(0, 0, 0),
//		cl::NDRange(m_width * m_height * sizeof(uint16_t) * m_max_disparity / 32),
//		cl::NDRange(256));
//
}

void StereoSGMVULKAN::matching_cost()
{
//	int MCOST_LINES128 = 2;
//	cl::Event matching_cost_event;
//	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_matching_cost_kernel_128, cl::NDRange(0, 0, 0),
//		cl::NDRange(128 * m_height / MCOST_LINES128, MCOST_LINES128),
//		cl::NDRange(128, MCOST_LINES128), nullptr, &matching_cost_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGMVULKAN::scan_cost()
{
//	//census_event.wait();
//	static const int PATHS_IN_BLOCK = 8;
//	cl::Event scan_cost_0_event;
//	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_horizontal_dir_kernel_0, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * m_height / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_0_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	//
//	//
//	cl::Event scan_cost_4_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_horizontal_dir_kernel_4, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * m_height / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_4_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	
//	cl::Event scan_cost_2_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_vertical_dir_kernel_2, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * m_width / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_2_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	
//	cl::Event scan_cost_6_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_vertical_dir_kernel_6, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * m_width / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_6_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//	const int obl_num_paths = m_width + m_height ;
//	cl::Event scan_cost_1_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_1, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_1_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	
//	cl::Event scan_cost_3_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_3, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_3_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	
//	
//	cl::Event scan_cost_5_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_5, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_5_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//	
//	cl::Event scan_cost_7_event;
//	m_err = m_command_queue.enqueueNDRangeKernel(m_compute_stereo_oblique_dir_kernel_7, cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * obl_num_paths / PATHS_IN_BLOCK, PATHS_IN_BLOCK),
//		cl::NDRange(32, PATHS_IN_BLOCK), nullptr, &scan_cost_7_event);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//
//
}

void StereoSGMVULKAN::winner_takes_all()
{
//	const int WTA_PIXEL_IN_BLOCK = 8;
//	cl::Event winner_takes_it_all;
//	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_winner_takes_all_kernel128,
//		cl::NDRange(0, 0, 0),
//		cl::NDRange(32 * m_width / WTA_PIXEL_IN_BLOCK, WTA_PIXEL_IN_BLOCK * m_height),
//		cl::NDRange(32, WTA_PIXEL_IN_BLOCK), nullptr, &winner_takes_it_all);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}

void StereoSGMVULKAN::median()
{
//	m_median_3x3.setArg(0, d_left_disparity);
//	m_median_3x3.setArg(1, d_tmp_left_disp);
//	cl::Event median_ev;
//	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_median_3x3,
//		cl::NDRange(0, 0, 0),
//		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
//		cl::NDRange(16, 16), nullptr, &median_ev);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
//
//
//	m_median_3x3.setArg(0, d_right_disparity);
//	m_median_3x3.setArg(1, d_tmp_right_disp);
//	m_err = m_command_queue.enqueueNDRangeKernel(m_median_3x3,
//		cl::NDRange(0, 0, 0),
//		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
//		cl::NDRange(16, 16), nullptr, &median_ev);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
//
}

void StereoSGMVULKAN::check_consistency_left()
{
//	cl::Event check_consistency_kernel_left_ev;
//	cl_int m_err = m_command_queue.enqueueNDRangeKernel(m_check_consistency_left,
//		cl::NDRange(0, 0, 0),
//		cl::NDRange((m_width + 16 - 1) / 16 * 16, (m_height + 16 - 1) / 16 * 16),
//		cl::NDRange(16, 16), nullptr, &check_consistency_kernel_left_ev);
//	checkErr(m_err, (std::to_string(__LINE__) + __FILE__).c_str());
}
