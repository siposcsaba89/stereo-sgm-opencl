/*
Copyright 2016 fixstars

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <libsgm_ocl/libsgm_ocl.h>
#include <iomanip>
#include <sstream>

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data);
std::tuple<cl_context, cl_device_id> initCLCTX(int platform_idx, int device_idx);


int main(int argc, char* argv[])
{
    std::string keys =
    "{ h help | | Print this message }"
    "{ @img_source_left | | Left images }"
    "{ @img_source_right |  | Right images }"
    "{ md max_disparity | 128 | Maximum disparity }"
    "{ sp subpixel | true | Compute subpixel accuracy }"
    "{ platform_idx | 0 | OpenCL plarform index }"
    "{ device_idx | 0 | OpenCL device index }"
    "{ np num_path | 4 | Num path to optimize, 4 or 8 }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }
    std::string left_filename_fmt, right_filename_fmt;
    left_filename_fmt = parser.get<std::string>(0);
    right_filename_fmt = parser.get<std::string>(1);
    int disp_size = parser.get<int>("max_disparity");

    cv::VideoCapture left_capture(left_filename_fmt);
    cv::VideoCapture right_capture(right_filename_fmt);
    if (!left_capture.isOpened())
    {
        std::cout << "Failed to open image stream: " << left_filename_fmt << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (!right_capture.isOpened())
    {
        std::cout << "Failed to open image stream: " << right_filename_fmt << std::endl;
        std::exit(EXIT_FAILURE);
    }
    left_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
    right_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);

    cv::Mat left, right;
    left_capture >> left;
    right_capture >> right;

    if (left.size() != right.size() || left.type() != right.type())
    {
        std::cerr << "mismatch input image size" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::Size img_size = left.size();

    int width = left.cols;
    int height = left.rows;

    if (width * height == 0)
    {
        std::cout << "Wrong input size: " << width << ", " << height << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cl_context cl_ctx;
    cl_device_id cl_device;
    int platform_idx = parser.get<int>("platform_idx");
    int device_idx = parser.get<int>("device_idx");
    std::tie(cl_ctx, cl_device) = initCLCTX(platform_idx, device_idx);
    cl_command_queue cl_queue = clCreateCommandQueue(cl_ctx, cl_device, 0, nullptr);

    sgm::cl::Parameters params;
    int input_depth = 8;
    params.subpixel = parser.get<bool>("subpixel");
    const int output_depth = (disp_size == 256 || params.subpixel) ? 16 : 8;
    params.path_type = parser.get<int>("num_path") == 8 ? sgm::cl::PathType::SCAN_8PATH : sgm::cl::PathType::SCAN_4PATH;
    params.uniqueness = 0.95f;

    {
        sgm::cl::StereoSGM<uint8_t> ssgm(width,
            height,
            disp_size,
            output_depth,
            cl_ctx,
            cl_device,
            params);

        cv::Mat img1c, img2c;

        bool should_close = false;
        int disp_type = output_depth == 8 ? CV_8UC1 : CV_16UC1;

        cv::Mat disp(img_size, disp_type), disp_color, disp_8u;
        cl_mem d_left, d_right, d_disp;
        d_left = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height, nullptr, nullptr);
        d_right = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height, nullptr, nullptr);
        d_disp = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height * output_depth / 8, nullptr, nullptr);

        while ((!should_close))
        {
            left_capture.read(img1c);
            if (img1c.empty())
            {
                std::cout << "Failed to read left image stream!" << std::endl;
                break;
            }
            right_capture.read(img2c);
            if (img2c.empty())
            {
                std::cout << "Failed to read right image stream!" << std::endl;
                break;
            }

            if (img1c.channels() != 1)
            {
                cv::cvtColor(img1c, left, cv::COLOR_BGR2GRAY);
                cv::cvtColor(img2c, right, cv::COLOR_BGR2GRAY);
            }
            else
            {
                left = img1c;
                right = img2c;
            }

            clEnqueueWriteBuffer(cl_queue, d_left, true, 0, width * height, left.data, 0, nullptr, nullptr);
            clEnqueueWriteBuffer(cl_queue, d_right, true, 0, width * height, right.data, 0, nullptr, nullptr);

            auto t = std::chrono::steady_clock::now();
            //ssgm.execute(left.data, right.data, reinterpret_cast<uint16_t*>(disp.data));
            ssgm.execute(d_left, d_right, d_disp);
            std::chrono::milliseconds dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t);
            clEnqueueReadBuffer(cl_queue, d_disp, true, 0, width * height * output_depth / 8, disp.data, 0, nullptr, nullptr);


            cv::Mat disparity_8u, disparity_color;
            disp.convertTo(disparity_8u, CV_8U, 255. / (disp_size * (params.subpixel ? 16 : 1)));
            cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
            const int invalid_disp = output_depth == 8
                ? static_cast<uint8_t>(ssgm.get_invalid_disparity())
                : static_cast<uint16_t>(ssgm.get_invalid_disparity());
            disparity_color.setTo(cv::Scalar(0, 0, 0), disp == invalid_disp);
            const int64_t fps = 1000 / dur.count();
            cv::putText(disparity_color, "sgm execution time: " + std::to_string(dur.count()) + "[msec] " + std::to_string(fps) + "[FPS]",
                cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));


            cv::imshow("left imagep", left);
            cv::imshow("disp", disparity_color);


            int key = cv::waitKey(1);
            if (key == 27)
            {
                should_close = true;
            }
        }
        clReleaseMemObject(d_left);
        clReleaseMemObject(d_right);
        clReleaseMemObject(d_disp);
    }
    clReleaseCommandQueue(cl_queue);
    clReleaseDevice(cl_device);
    clReleaseContext(cl_ctx);
    return 0;
}


std::tuple<cl_context, cl_device_id> initCLCTX(int platform_idx, int device_idx)
{
    cl_uint num_platform;
    clGetPlatformIDs(0, nullptr, &num_platform);
    assert((size_t)platform_idx < num_platform);
    std::vector<cl_platform_id> platform_ids(num_platform);
    clGetPlatformIDs(num_platform, platform_ids.data(), nullptr);
    if(platform_ids.size() <= platform_idx)
    {
        std::cout << "Wrong platform index!" << std::endl;
        exit(0);
    }
    cl_uint num_devices;
    clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    assert((size_t)device_idx < num_devices);
    std::vector<cl_device_id> cl_devices(num_devices);
    clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_GPU, num_devices, cl_devices.data(), nullptr);
    cl_device_id cl_device = cl_devices[device_idx];
    cl_int err;
    cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_devices[device_idx], context_error_callback, NULL, &err);

    if (err != CL_SUCCESS)
    {
        std::cout << "Error creating context " << err << std::endl;
        throw std::runtime_error("Error creating context!");
    }
    {
        size_t name_size_in_bytes;
        clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_NAME, 0, nullptr, &name_size_in_bytes);
        std::string platform_name;
        platform_name.resize(name_size_in_bytes);
        clGetPlatformInfo(platform_ids[platform_idx], CL_PLATFORM_NAME,
            platform_name.size(),
            (void*)platform_name.data(), nullptr);
        std::cout << "Platform name: " << platform_name << std::endl;
    }
    {
        size_t name_size_in_bytes;
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME, 0, nullptr, &name_size_in_bytes);
        std::string dev_name;
        dev_name.resize(name_size_in_bytes);
        clGetDeviceInfo(cl_devices[device_idx], CL_DEVICE_NAME,
            dev_name.size(),
            (void*)dev_name.data(), nullptr);
        std::cout << "Device name: " << dev_name << std::endl;
    }
    return std::make_tuple(cl_ctx, cl_device);
}

void context_error_callback(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
    std::cout << "opencl error : " << errinfo << std::endl;
}
