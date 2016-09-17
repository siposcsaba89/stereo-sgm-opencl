#include "stereo-sgm.h"
#include <vector>
#include <assert.h>

int platform_index = 1;
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

StereoSGM::StereoSGM()
{

	std::vector<cl::Platform> platform_list;
	cl::Platform::get(&platform_list);

	assert(platform_index < platform_list.size());

	std::string platformVendor;
	std::string platformVersion;

	platform_list[platform_index].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
	printf("Platform vendor: %s \n" ,platformVendor.c_str());

	platform_list[platform_index].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVersion);
	printf("Platform version: %s \n", platformVersion.c_str());

	std::vector<cl::Device> devices;
	cl_int m_err = platform_list[platform_index].getDevices(
		CL_DEVICE_TYPE_ALL, &devices);
	checkErr(m_err, "getDevicesList");

	assert(devices.size() > device_index);
	std::string device_name;
	devices[device_index].getInfo(CL_DEVICE_NAME, &device_name);
	printf("Selected device: %s \n", device_name.c_str());

}

StereoSGM::~StereoSGM()
{
}
