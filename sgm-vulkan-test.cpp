///*
//Copyright 2016 fixstars
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http ://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//*/
//#include <iomanip>
//#include <sstream>
//#include <stdlib.h>
//#include <iostream>
//#include <string>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/calib3d/calib3d.hpp"
//#include "sgm-vulkan/sgm-vulkan.h"
//
//static void saveXYZ(const char* filename, const cv::Mat& mat)
//{
//	const double max_z = 1.0e4;
//	FILE* fp = fopen(filename, "wt");
//	for (int y = 0; y < mat.rows; y++)
//	{
//		for (int x = 0; x < mat.cols; x++)
//		{
//			cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
//			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
//			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
//		}
//	}
//	fclose(fp);
//}
//
//
//int main(int argc, char* argv[]) {
//
//	// imgleft%2509d.pgm imgright%2509d.pgm
//	// C:\cv_data\2010_03_09_drive_0019_pgm\I1_%2506d.pgm C:\cv_data\2010_03_09_drive_0019_pgm\I2_%2506d.pgm 128 370
//	// C:\cv_data\2010_03_09_drive_0051_pgm\I1_%2506d.pgm C:\cv_data\2010_03_09_drive_0051_pgm\I2_%2506d.pgm 128 400
//
//	// C:\cv_data\arpadhid\%2504d_left.pgm C:\cv_data\arpadhid\%2504d_right.pgm 64
//
//	//r:\2016.09.20_stereo_montevideo\video110414137\%05d_img.png r:\2016.09.20_stereo_montevideo\video210414137\%2505d_img.png
//
//	if (argc < 3) {
//		std::cerr << "usage: StereoSGMCL left_img_fmt right_img_fmt [disp_size] [max_frame_num]" << std::endl;
//		std::exit(EXIT_FAILURE);
//	}
//	std::string left_filename_fmt, right_filename_fmt;
//	left_filename_fmt = argv[1];
//	right_filename_fmt = argv[2];
//
//	cv::VideoCapture left_capture(left_filename_fmt);
//	cv::VideoCapture right_capture(right_filename_fmt);
//	left_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
//	right_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
//	// dangerous
//	/*char buf[1024];
//	sprintf(buf, left_filename_fmt.c_str(), 0);*/
//	cv::Mat left, right;// = cv::imread(buf, -1);
//						/*sprintf(buf, right_filename_fmt.c_str(), 0);
//						cv::Mat right = cv::imread(buf, -1);*/
//	left_capture >> left;
//	right_capture >> right;
//
//	int disp_size = 128;
//	if (argc >= 4) {
//		disp_size = atoi(argv[3]);
//	}
//
//	int max_frame = 100;
//	if (argc >= 5) {
//		max_frame = atoi(argv[4]);
//	}
//
//
//	if (left.size() != right.size() || left.type() != right.type()) {
//		std::cerr << "mismatch input image size" << std::endl;
//		std::exit(EXIT_FAILURE);
//	}
//
//	cv::Size img_size = left.size();
//
//	std::string extrinsic_filename = "d:/extrinsics.yml";
//	std::string intrinsic_filename = "d:/intrinsics.yml";
//
//	cv::Rect roi1, roi2;
//	cv::Mat Q;
//	cv::Mat map11, map12, map21, map22;
//	cv::Mat R, T, R1, P1, R2, P2;
//	cv::Mat M1, D1, M2, D2;
//	if (!intrinsic_filename.empty())
//	{
//		// reading intrinsic parameters
//		cv::FileStorage fs(intrinsic_filename, cv::FileStorage::READ);
//		if (!fs.isOpened())
//		{
//			printf("Failed to open file %s\n", intrinsic_filename.c_str());
//			return -1;
//		}
//
//		fs["M1"] >> M1;
//		fs["D1"] >> D1;
//		fs["M2"] >> M2;
//		fs["D2"] >> D2;
//
//		double scale = 1.0;
//
//		M1 *= scale;
//		M2 *= scale;
//
//		fs.open(extrinsic_filename, cv::FileStorage::READ);
//		if (!fs.isOpened())
//		{
//			printf("Failed to open file %s\n", extrinsic_filename.c_str());
//			return -1;
//		}
//
//		fs["R"] >> R;
//		fs["T"] >> T;
//
//		cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
//
//
//		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
//		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
//
//	}
//
//
//
//	int bits = 8;
//
//	//switch (left.type()) {
//	//case CV_8UC1: bits = 8; break;
//	//case CV_16UC1: bits = 16; break;
//	//default:
//	//	std::cerr << "invalid input image color format" << left.type() << std::endl;
//	//	std::exit(EXIT_FAILURE);
//	//}
//
//	int width = left.cols;
//	int height = left.rows;
//
//
//	
//	float fl = (float)M1.at<double>(0);
//	float cx = (float)M1.at<double>(2);
//	float cy = (float)M1.at<double>(5);
//	float b_d = (float)cv::norm(T, cv::NORM_L2);
//
//	StereoSGMVULKAN ssgm(width, height, disp_size);// , bits, 16, fl, cx, cy, b_d);
//
//	uint16_t* d_output_buffer = nullptr;
//
//	cv::Mat img1c, img2c;
//	cv::Mat img1r, img2r;
//
//	int frame_no = 0;
//	bool should_close = false;
//
//    while ((!should_close && left_capture.read(img1c) && right_capture.read(img2c))) {
//
//		if (frame_no == max_frame) { frame_no = 0; }
//
//		//sprintf(buf, left_filename_fmt.c_str(), frame_no);
//		//cv::Mat left = cv::imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
//		//sprintf(buf, right_filename_fmt.c_str(), frame_no);
//		//cv::Mat right = cv::imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
//
//		cv::cvtColor(img1c, left, CV_BGR2GRAY);
//		cv::cvtColor(img2c, right, CV_BGR2GRAY);
//		//cv::cvtColor(img1c, left, CV_BayerRG2GRAY);
//		//cv::cvtColor(img2c, right, CV_BayerRG2GRAY);
//
//
//		cv::remap(left, img1r, map11, map12, cv::INTER_LINEAR);
//		cv::remap(right, img2r, map21, map22, cv::INTER_LINEAR);
//
//		left = img1r;
//		right = img2r;
//
//
//
//		if (left.size() == cv::Size(0, 0) || right.size() == cv::Size(0, 0)) {
//			max_frame = frame_no;
//			frame_no = 0;
//			continue;
//		}
//
////		cv::Mat v_disp(img_size.height, disp_size * 2, CV_32S);
////		cv::Mat u_disp(disp_size, img_size.width, CV_32S);
////		cv::Mat cu_disp(img_size, CV_32S);
//		clock_t st = clock();
////		std::vector<uint32_t> v_disp_road(img_size.height, 0);
////		std::vector<uint32_t> free_space(img_size.width, 0);
//
//
////		cv::Mat free_space_voting_res(img_size.height, img_size.width, CV_32FC1);
//
////		ssgm.execute(left.data, right.data, (void**)&d_output_buffer, v_disp.data, v_disp_road.data(), u_disp.data, cu_disp.data,
////			free_space.data(), free_space_voting_res.data); // , sgm::DST_TYPE_CUDA_PTR, 16);
//		static cv::Mat disp(img_size, CV_16UC1);
//
//		ssgm.execute(left.data, right.data, disp.data); // , sgm::DST_TYPE_CUDA_PTR, 16);
//		std::cout << clock() - st << std::endl;
//
//		cv::imshow("disp", (disp * 2) * 256);
//
//		int key = cv::waitKey(1);
//
//
//
//		switch (key) {
//		case 27:
//		{
//			should_close = true;
//		}
//		break;
//		case 1:
//		{
//			//renderer.render_disparity(d_output_buffer, disp_size);
//			//static int fc = 0;
//			//++fc;
//			//if (fc % 100 == 0)
//
//			//cv::Mat cd(img_size, CV_8UC3);
//			//glReadPixels(0, 0, img_size.width, img_size.height, GL_BGR, GL_UNSIGNED_BYTE, cd.data);
//			//cv::flip(cd, cd, 0);
//			//cv::Mat canny_out;
//			//cv::Canny(cd, canny_out, 25, 50, 3);;
//			//
//			//
//			//cv::imshow("objs", canny_out);
//			//std::stringstream ss;
//			//ss << std::setw(4) << std::setfill('0') << fc;
//			//cv::imwrite("d:/disp/" + ss.str() + ".png", cd);
//		}
//		break;
//		case 2:
//			//renderer.render_disparity_color(d_output_buffer, disp_size);
//			//static int fc = 0;
//			//++fc;
//			////if (fc % 100 == 0)
//			//{
//			//	cv::Mat cd(img_size, CV_8UC3);
//			//	glReadPixels(0, 0, img_size.width, img_size.height, GL_BGR, GL_UNSIGNED_BYTE, cd.data);
//			//	cv::flip(cd, cd, 0);
//			//	cv::Mat canny_out;
//			//	cv::Canny(cd, canny_out, 100, 100, 3);;
//			//
//			//
//			//	cv::imshow("objs", canny_out);
//			//	//std::stringstream ss;
//			//	//ss << std::setw(4) << std::setfill('0') << fc;
//			//	//cv::imwrite("d:/disp/" + ss.str() + ".png", cd);
//			//}
//			break;
//		}
//
//		//Polygon p;
//		//p.polyg = {
//		//	Point2f(600, 450),
//		//	Point2f(400, 680),
//		//	Point2f(1100, 680),
//		//	Point2f(850, 450)
//		//};
//
//
//		//std::vector<Polygon> objects;
//
//		//bool area_clear = false;// simple_blov_det.compute((uint16_t*)d_output_buffer, img_size.width, img_size.height, objects, p);
//
//		//if (!area_clear)
//		//{
//		//	std::cout << "WAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRNNNNNNNNNNIIIIIIIIIINNNNNNNNNNGGGGGGGGGG \n \n";
//		//}
//		//else
//		//{
//		//	std::cout << "gaaaaaaaaaazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzztttttttttttttttttttttttttttttttt \n \n";
//		//}
//
//
//		//cv::Mat obj_img = left;// (img_size, CV_8UC1);
//		//
//		//for (auto & obj : objects)
//		//{
//		//	for (int i = 1; i < obj.polyg.size(); ++i)
//		//	{
//		//		cv::line(obj_img, cv::Point(obj.polyg[i - 1].x(), obj.polyg[i - 1].y()), cv::Point(obj.polyg[i].x(), obj.polyg[i].y()),
//		//			256, 3);
//		//	}
//		//}
//		//
//		//
//		//for (int i = 1; i < p.polyg.size(); ++i)
//		//{
//		//	cv::line(obj_img, cv::Point2f(p.polyg[i-1].x(), p.polyg[i - 1].y()), cv::Point2f(p.polyg[i].x(), p.polyg[i].y()), 255, 5);
//		//}
//
//
//
//
//
//		//demo.swap_buffer();
//		frame_no++;
//
//		//for (int i = 0; i < img_size.height; ++i)
//		//	cv::circle(v_disp, cv::Point2f(v_disp_road[i], i), 1, 128, 2);
//
//		//for (int i = 0; i < height / 16; ++i)
//		//{
//		//	cv::circle(cu_disp, cv::Point(250, height - i * 16 - 1), 3, 255, 1);
//		//	cv::putText(cu_disp, std::to_string(i), cv::Point(270, height - i * 16 - 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, 255);
//		//	cv::line(cu_disp, cv::Point(300, height - i * 16 - 1), cv::Point(1200, height - i * 16 - 1), 10, 1);
//		//}
//
//		//for (int i = 0; i < width; ++i)
//		//{
//		//	cv::circle(u_disp, cv::Point(i, free_space[i]), 3, 255, 1);
//		//}
//		//cv::imshow("v_disp", u_disp * 1024);
//		//
//		//cv::imshow("free_space_voting_res", free_space_voting_res);
//		//
//		//cv::waitKey(1);
//
//	}
//
//	delete d_output_buffer;
//    return 0;
//}






















// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>

#define BAIL_ON_BAD_RESULT(result) \
  if (VK_SUCCESS != (result)) { fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(-1); }

VkResult vkGetBestTransferQueueNPH(VkPhysicalDevice physicalDevice, uint32_t* queueFamilyIndex) {
    uint32_t queueFamilyPropertiesCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

    VkQueueFamilyProperties* const queueFamilyProperties = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties);

    // first try and find a queue that has just the transfer bit set
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!)
        const VkQueueFlags maskedFlags = (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

        if (!((VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) & maskedFlags) &&
            (VK_QUEUE_TRANSFER_BIT & maskedFlags)) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    // otherwise we'll prefer using a compute-only queue,
    // remember that having compute on the queue implicitly enables transfer!
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!)
        const VkQueueFlags maskedFlags = (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

        if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    // lastly get any queue that'll work for us (graphics, compute or transfer bit set)
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!)
        const VkQueueFlags maskedFlags = (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

        if ((VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT) & maskedFlags) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    return VK_ERROR_INITIALIZATION_FAILED;
}

VkResult vkGetBestComputeQueueNPH(VkPhysicalDevice physicalDevice, uint32_t* queueFamilyIndex) {
    uint32_t queueFamilyPropertiesCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

    VkQueueFamilyProperties* const queueFamilyProperties = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties);

    // first try and find a queue that has just the compute bit set
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
        const VkQueueFlags maskedFlags = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
            queueFamilyProperties[i].queueFlags);

        if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    // lastly get any queue that'll work for us
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
        const VkQueueFlags maskedFlags = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
            queueFamilyProperties[i].queueFlags);

        if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    return VK_ERROR_INITIALIZATION_FAILED;
}

int main(int argc, const char * const argv[]) {
    (void)argc;
    (void)argv;

    const VkApplicationInfo applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        0,
        "VKComputeSample",
        0,
        "",
        0,
        VK_MAKE_VERSION(1, 0, 9)
    };

    const VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        0,
        0,
        &applicationInfo,
        0,
        0,
        0,
        0
    };

    VkInstance instance;
    BAIL_ON_BAD_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &instance));

    uint32_t physicalDeviceCount = 0;
    BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0));

    VkPhysicalDevice* const physicalDevices = (VkPhysicalDevice*)malloc(
        sizeof(VkPhysicalDevice) * physicalDeviceCount);

    BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices));

    for (uint32_t i = 0; i < physicalDeviceCount; i++) {
        uint32_t queueFamilyIndex = 0;
        BAIL_ON_BAD_RESULT(vkGetBestComputeQueueNPH(physicalDevices[i], &queueFamilyIndex));

        const float queuePrioritory = 1.0f;
        const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            0,
            0,
            queueFamilyIndex,
            1,
            &queuePrioritory
        };

        const VkDeviceCreateInfo deviceCreateInfo = {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            0,
            0,
            1,
            &deviceQueueCreateInfo,
            0,
            0,
            0,
            0,
            0
        };

        VkDevice device;
        BAIL_ON_BAD_RESULT(vkCreateDevice(physicalDevices[i], &deviceCreateInfo, 0, &device));

        VkPhysicalDeviceMemoryProperties properties;

        vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &properties);

        const int32_t bufferLength = 16384;

        const uint32_t bufferSize = sizeof(int32_t) * bufferLength;

        // we are going to need two buffers from this one memory
        const VkDeviceSize memorySize = bufferSize * 2;

        // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
        uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

        for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
            if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[k].propertyFlags) &&
                (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[k].propertyFlags) &&
                (memorySize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size)) {
                memoryTypeIndex = k;
                break;
            }
        }

        BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

        const VkMemoryAllocateInfo memoryAllocateInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            0,
            memorySize,
            memoryTypeIndex
        };

        VkDeviceMemory memory;
        BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory));

        int32_t *payload;
        BAIL_ON_BAD_RESULT(vkMapMemory(device, memory, 0, memorySize, 0, (void **)&payload));

        for (uint32_t k = 1; k < memorySize / sizeof(int32_t); k++) {
            payload[k] = rand();
        }

        vkUnmapMemory(device, memory);

        const VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            0,
            0,
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &queueFamilyIndex
        };

        VkBuffer in_buffer;
        BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &in_buffer));

        BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, in_buffer, memory, 0));

        VkBuffer out_buffer;
        BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &out_buffer));

        BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, out_buffer, memory, bufferSize));

        enum {
            RESERVED_ID = 0,
            FUNC_ID,
            IN_ID,
            OUT_ID,
            GLOBAL_INVOCATION_ID,
            VOID_TYPE_ID,
            FUNC_TYPE_ID,
            INT_TYPE_ID,
            INT_ARRAY_TYPE_ID,
            STRUCT_ID,
            POINTER_TYPE_ID,
            ELEMENT_POINTER_TYPE_ID,
            INT_VECTOR_TYPE_ID,
            INT_VECTOR_POINTER_TYPE_ID,
            INT_POINTER_TYPE_ID,
            CONSTANT_ZERO_ID,
            CONSTANT_ARRAY_LENGTH_ID,
            LABEL_ID,
            IN_ELEMENT_ID,
            OUT_ELEMENT_ID,
            GLOBAL_INVOCATION_X_ID,
            GLOBAL_INVOCATION_X_PTR_ID,
            TEMP_LOADED_ID,
            BOUND
        };

        enum {
            INPUT = 1,
            UNIFORM = 2,
            BUFFER_BLOCK = 3,
            ARRAY_STRIDE = 6,
            BUILTIN = 11,
            BINDING = 33,
            OFFSET = 35,
            DESCRIPTOR_SET = 34,
            GLOBAL_INVOCATION = 28,
            OP_TYPE_VOID = 19,
            OP_TYPE_FUNCTION = 33,
            OP_TYPE_INT = 21,
            OP_TYPE_VECTOR = 23,
            OP_TYPE_ARRAY = 28,
            OP_TYPE_STRUCT = 30,
            OP_TYPE_POINTER = 32,
            OP_VARIABLE = 59,
            OP_DECORATE = 71,
            OP_MEMBER_DECORATE = 72,
            OP_FUNCTION = 54,
            OP_LABEL = 248,
            OP_ACCESS_CHAIN = 65,
            OP_CONSTANT = 43,
            OP_LOAD = 61,
            OP_STORE = 62,
            OP_RETURN = 253,
            OP_FUNCTION_END = 56,
            OP_CAPABILITY = 17,
            OP_MEMORY_MODEL = 14,
            OP_ENTRY_POINT = 15,
            OP_EXECUTION_MODE = 16,
            OP_COMPOSITE_EXTRACT = 81,
        };

        int32_t shader[] = {
            // first is the SPIR-V header
            0x07230203, // magic header ID
            0x00010000, // version 1.0.0
            0,          // generator (optional)
            BOUND,      // bound
            0,          // schema

                        // OpCapability Shader
                        (2 << 16) | OP_CAPABILITY, 1,

                        // OpMemoryModel Logical Simple
                        (3 << 16) | OP_MEMORY_MODEL, 0, 0,

                        // OpEntryPoint GLCompute %FUNC_ID "f" %IN_ID %OUT_ID
                        (4 << 16) | OP_ENTRY_POINT, 5, FUNC_ID, 0x00000066,

                        // OpExecutionMode %FUNC_ID LocalSize 1 1 1
                        (6 << 16) | OP_EXECUTION_MODE, FUNC_ID, 17, 1, 1, 1,

                        // next declare decorations

                        (3 << 16) | OP_DECORATE, STRUCT_ID, BUFFER_BLOCK,

                        (4 << 16) | OP_DECORATE, GLOBAL_INVOCATION_ID, BUILTIN, GLOBAL_INVOCATION,

                        (4 << 16) | OP_DECORATE, IN_ID, DESCRIPTOR_SET, 0,

                        (4 << 16) | OP_DECORATE, IN_ID, BINDING, 0,

                        (4 << 16) | OP_DECORATE, OUT_ID, DESCRIPTOR_SET, 0,

                        (4 << 16) | OP_DECORATE, OUT_ID, BINDING, 1,

                        (4 << 16) | OP_DECORATE, INT_ARRAY_TYPE_ID, ARRAY_STRIDE, 4,

                        (5 << 16) | OP_MEMBER_DECORATE, STRUCT_ID, 0, OFFSET, 0,

                        // next declare types
                        (2 << 16) | OP_TYPE_VOID, VOID_TYPE_ID,

                        (3 << 16) | OP_TYPE_FUNCTION, FUNC_TYPE_ID, VOID_TYPE_ID,

                        (4 << 16) | OP_TYPE_INT, INT_TYPE_ID, 32, 1,

                        (4 << 16) | OP_CONSTANT, INT_TYPE_ID, CONSTANT_ARRAY_LENGTH_ID, bufferLength,

                        (4 << 16) | OP_TYPE_ARRAY, INT_ARRAY_TYPE_ID, INT_TYPE_ID, CONSTANT_ARRAY_LENGTH_ID,

                        (3 << 16) | OP_TYPE_STRUCT, STRUCT_ID, INT_ARRAY_TYPE_ID,

                        (4 << 16) | OP_TYPE_POINTER, POINTER_TYPE_ID, UNIFORM, STRUCT_ID,

                        (4 << 16) | OP_TYPE_POINTER, ELEMENT_POINTER_TYPE_ID, UNIFORM, INT_TYPE_ID,

                        (4 << 16) | OP_TYPE_VECTOR, INT_VECTOR_TYPE_ID, INT_TYPE_ID, 3,

                        (4 << 16) | OP_TYPE_POINTER, INT_VECTOR_POINTER_TYPE_ID, INPUT, INT_VECTOR_TYPE_ID,

                        (4 << 16) | OP_TYPE_POINTER, INT_POINTER_TYPE_ID, INPUT, INT_TYPE_ID,

                        // then declare constants
                        (4 << 16) | OP_CONSTANT, INT_TYPE_ID, CONSTANT_ZERO_ID, 0,

                        // then declare variables
                        (4 << 16) | OP_VARIABLE, POINTER_TYPE_ID, IN_ID, UNIFORM,

                        (4 << 16) | OP_VARIABLE, POINTER_TYPE_ID, OUT_ID, UNIFORM,

                        (4 << 16) | OP_VARIABLE, INT_VECTOR_POINTER_TYPE_ID, GLOBAL_INVOCATION_ID, INPUT,

                        // then declare function
                        (5 << 16) | OP_FUNCTION, VOID_TYPE_ID, FUNC_ID, 0, FUNC_TYPE_ID,

                        (2 << 16) | OP_LABEL, LABEL_ID,

                        (5 << 16) | OP_ACCESS_CHAIN, INT_POINTER_TYPE_ID, GLOBAL_INVOCATION_X_PTR_ID, GLOBAL_INVOCATION_ID, CONSTANT_ZERO_ID,

                        (4 << 16) | OP_LOAD, INT_TYPE_ID, GLOBAL_INVOCATION_X_ID, GLOBAL_INVOCATION_X_PTR_ID,

                        (6 << 16) | OP_ACCESS_CHAIN, ELEMENT_POINTER_TYPE_ID, IN_ELEMENT_ID, IN_ID, CONSTANT_ZERO_ID, GLOBAL_INVOCATION_X_ID,

                        (4 << 16) | OP_LOAD, INT_TYPE_ID, TEMP_LOADED_ID, IN_ELEMENT_ID,

                        (6 << 16) | OP_ACCESS_CHAIN, ELEMENT_POINTER_TYPE_ID, OUT_ELEMENT_ID, OUT_ID, CONSTANT_ZERO_ID, GLOBAL_INVOCATION_X_ID,

                        (3 << 16) | OP_STORE, OUT_ELEMENT_ID, TEMP_LOADED_ID,

                        (1 << 16) | OP_RETURN,

                        (1 << 16) | OP_FUNCTION_END,
        };

        VkShaderModuleCreateInfo shaderModuleCreateInfo = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            0,
            0,
            sizeof(shader),
            (uint32_t*)shader
        };

        VkShaderModule shader_module;

        BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_module));

        VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2] = {
            {
                0,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0
            },
            {
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0
            }
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            0,
            0,
            2,
            descriptorSetLayoutBindings
        };

        VkDescriptorSetLayout descriptorSetLayout;
        BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0, &descriptorSetLayout));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            0,
            0,
            1,
            &descriptorSetLayout,
            0,
            0
        };

        VkPipelineLayout pipelineLayout;
        BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, &pipelineLayout));

        VkComputePipelineCreateInfo computePipelineCreateInfo = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            0,
            0,
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                0,
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                shader_module,
                "f",
                0
            },
            pipelineLayout,
            0,
            0
        };

        VkPipeline pipeline;
        BAIL_ON_BAD_RESULT(vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo, 0, &pipeline));

        VkCommandPoolCreateInfo commandPoolCreateInfo = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            0,
            0,
            queueFamilyIndex
        };

        VkDescriptorPoolSize descriptorPoolSize = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            2
        };

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            0,
            0,
            1,
            1,
            &descriptorPoolSize
        };

        VkDescriptorPool descriptorPool;
        BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, 0, &descriptorPool));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            0,
            descriptorPool,
            1,
            &descriptorSetLayout
        };

        VkDescriptorSet descriptorSet;
        BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

        VkDescriptorBufferInfo in_descriptorBufferInfo = {
            in_buffer,
            0,
            VK_WHOLE_SIZE
        };

        VkDescriptorBufferInfo out_descriptorBufferInfo = {
            out_buffer,
            0,
            VK_WHOLE_SIZE
        };

        VkWriteDescriptorSet writeDescriptorSet[2] = {
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                0,
                descriptorSet,
                0,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0,
                &in_descriptorBufferInfo,
                0
            },
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                0,
                descriptorSet,
                1,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0,
                &out_descriptorBufferInfo,
                0
            }
        };

        vkUpdateDescriptorSets(device, 2, writeDescriptorSet, 0, 0);

        VkCommandPool commandPool;
        BAIL_ON_BAD_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            0,
            commandPool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1
        };

        VkCommandBuffer commandBuffer;
        BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

        VkCommandBufferBeginInfo commandBufferBeginInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            0,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            0
        };

        BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout, 0, 1, &descriptorSet, 0, 0);

        vkCmdDispatch(commandBuffer, bufferSize / sizeof(int32_t), 1, 1);

        BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));

        VkQueue queue;
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            0,
            0,
            0,
            0,
            1,
            &commandBuffer,
            0,
            0
        };

        BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));

        BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));

        BAIL_ON_BAD_RESULT(vkMapMemory(device, memory, 0, memorySize, 0, (void **)&payload));

        for (uint32_t k = 0, e = bufferSize / sizeof(int32_t); k < e; k++) {
            BAIL_ON_BAD_RESULT(payload[k + e] == payload[k] ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY);
        }
    }
}