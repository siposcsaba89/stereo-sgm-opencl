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
#include "stereo-sgm.h"

static void saveXYZ(const char* filename, const cv::Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

#include <iomanip>
#include <sstream>
int main(int argc, char* argv[]) {

	// imgleft%2509d.pgm imgright%2509d.pgm
	// C:\cv_data\2010_03_09_drive_0019_pgm\I1_%2506d.pgm C:\cv_data\2010_03_09_drive_0019_pgm\I2_%2506d.pgm 128 370
	// C:\cv_data\2010_03_09_drive_0051_pgm\I1_%2506d.pgm C:\cv_data\2010_03_09_drive_0051_pgm\I2_%2506d.pgm 128 400

	// C:\cv_data\arpadhid\%2504d_left.pgm C:\cv_data\arpadhid\%2504d_right.pgm 64

	//r:\2016.09.20_stereo_montevideo\video110414137\%05d_img.png r:\2016.09.20_stereo_montevideo\video210414137\%2505d_img.png

	if (argc < 3) {
		std::cerr << "usage: stereosgm left_img_fmt right_img_fmt [disp_size] [max_frame_num]" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::string left_filename_fmt, right_filename_fmt;
	left_filename_fmt = argv[1];
	right_filename_fmt = argv[2];

	cv::VideoCapture left_capture(left_filename_fmt);
	cv::VideoCapture right_capture(right_filename_fmt);
	left_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
	right_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
	// dangerous
	/*char buf[1024];
	sprintf(buf, left_filename_fmt.c_str(), 0);*/
	cv::Mat left, right;// = cv::imread(buf, -1);
						/*sprintf(buf, right_filename_fmt.c_str(), 0);
						cv::Mat right = cv::imread(buf, -1);*/
	left_capture >> left;
	right_capture >> right;

	int disp_size = 128;
	if (argc >= 4) {
		disp_size = atoi(argv[3]);
	}

	int max_frame = 100;
	if (argc >= 5) {
		max_frame = atoi(argv[4]);
	}


	if (left.size() != right.size() || left.type() != right.type()) {
		std::cerr << "mismatch input image size" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cv::Size img_size = left.size();

	std::string extrinsic_filename = "d:/extrinsics.yml";
	std::string intrinsic_filename = "d:/intrinsics.yml";

	cv::Rect roi1, roi2;
	cv::Mat Q;
	cv::Mat map11, map12, map21, map22;
	cv::Mat R, T, R1, P1, R2, P2;
	cv::Mat M1, D1, M2, D2;
	if (!intrinsic_filename.empty())
	{
		// reading intrinsic parameters
		cv::FileStorage fs(intrinsic_filename, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename.c_str());
			return -1;
		}

		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		double scale = 1.0;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename.c_str());
			return -1;
		}

		fs["R"] >> R;
		fs["T"] >> T;

		cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);


		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	}



	int bits = 8;

	//switch (left.type()) {
	//case CV_8UC1: bits = 8; break;
	//case CV_16UC1: bits = 16; break;
	//default:
	//	std::cerr << "invalid input image color format" << left.type() << std::endl;
	//	std::exit(EXIT_FAILURE);
	//}

	int width = left.cols;
	int height = left.rows;


	
	float fl = (float)M1.at<double>(0);
	float cx = (float)M1.at<double>(2);
	float cy = (float)M1.at<double>(5);
	float b_d = (float)cv::norm(T, cv::NORM_L2);

	StereoSGM ssgm(width, height, disp_size);// , bits, 16, fl, cx, cy, b_d);

	uint16_t* d_output_buffer = nullptr;

	cv::Mat img1c, img2c;
	cv::Mat img1r, img2r;

	int frame_no = 0;
	bool should_close = false;

    while ((!should_close && left_capture.read(img1c) && right_capture.read(img2c))) {

		if (frame_no == max_frame) { frame_no = 0; }

		//sprintf(buf, left_filename_fmt.c_str(), frame_no);
		//cv::Mat left = cv::imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
		//sprintf(buf, right_filename_fmt.c_str(), frame_no);
		//cv::Mat right = cv::imread(buf, CV_LOAD_IMAGE_GRAYSCALE);

		cv::cvtColor(img1c, left, CV_BGR2GRAY);
		cv::cvtColor(img2c, right, CV_BGR2GRAY);
		//cv::cvtColor(img1c, left, CV_BayerRG2GRAY);
		//cv::cvtColor(img2c, right, CV_BayerRG2GRAY);


		cv::remap(left, img1r, map11, map12, cv::INTER_LINEAR);
		cv::remap(right, img2r, map21, map22, cv::INTER_LINEAR);

		left = img1r;
		right = img2r;



		if (left.size() == cv::Size(0, 0) || right.size() == cv::Size(0, 0)) {
			max_frame = frame_no;
			frame_no = 0;
			continue;
		}

//		cv::Mat v_disp(img_size.height, disp_size * 2, CV_32S);
//		cv::Mat u_disp(disp_size, img_size.width, CV_32S);
//		cv::Mat cu_disp(img_size, CV_32S);
		clock_t st = clock();
//		std::vector<uint32_t> v_disp_road(img_size.height, 0);
//		std::vector<uint32_t> free_space(img_size.width, 0);


//		cv::Mat free_space_voting_res(img_size.height, img_size.width, CV_32FC1);

//		ssgm.execute(left.data, right.data, (void**)&d_output_buffer, v_disp.data, v_disp_road.data(), u_disp.data, cu_disp.data,
//			free_space.data(), free_space_voting_res.data); // , sgm::DST_TYPE_CUDA_PTR, 16);
		static cv::Mat disp(img_size, CV_16UC1);

		ssgm.execute(left.data, right.data, disp.data); // , sgm::DST_TYPE_CUDA_PTR, 16);
		std::cout << clock() - st << std::endl;

		cv::imshow("disp", (disp * 2) * 256);

		int key = cv::waitKey(1);



		switch (key) {
		case 27:
		{
			should_close = true;
		}
		break;
		case 1:
		{
			//renderer.render_disparity(d_output_buffer, disp_size);
			//static int fc = 0;
			//++fc;
			//if (fc % 100 == 0)

			//cv::Mat cd(img_size, CV_8UC3);
			//glReadPixels(0, 0, img_size.width, img_size.height, GL_BGR, GL_UNSIGNED_BYTE, cd.data);
			//cv::flip(cd, cd, 0);
			//cv::Mat canny_out;
			//cv::Canny(cd, canny_out, 25, 50, 3);;
			//
			//
			//cv::imshow("objs", canny_out);
			//std::stringstream ss;
			//ss << std::setw(4) << std::setfill('0') << fc;
			//cv::imwrite("d:/disp/" + ss.str() + ".png", cd);
		}
		break;
		case 2:
			//renderer.render_disparity_color(d_output_buffer, disp_size);
			//static int fc = 0;
			//++fc;
			////if (fc % 100 == 0)
			//{
			//	cv::Mat cd(img_size, CV_8UC3);
			//	glReadPixels(0, 0, img_size.width, img_size.height, GL_BGR, GL_UNSIGNED_BYTE, cd.data);
			//	cv::flip(cd, cd, 0);
			//	cv::Mat canny_out;
			//	cv::Canny(cd, canny_out, 100, 100, 3);;
			//
			//
			//	cv::imshow("objs", canny_out);
			//	//std::stringstream ss;
			//	//ss << std::setw(4) << std::setfill('0') << fc;
			//	//cv::imwrite("d:/disp/" + ss.str() + ".png", cd);
			//}
			break;
		}

		//Polygon p;
		//p.polyg = {
		//	Point2f(600, 450),
		//	Point2f(400, 680),
		//	Point2f(1100, 680),
		//	Point2f(850, 450)
		//};


		//std::vector<Polygon> objects;

		//bool area_clear = false;// simple_blov_det.compute((uint16_t*)d_output_buffer, img_size.width, img_size.height, objects, p);

		//if (!area_clear)
		//{
		//	std::cout << "WAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRNNNNNNNNNNIIIIIIIIIINNNNNNNNNNGGGGGGGGGG \n \n";
		//}
		//else
		//{
		//	std::cout << "gaaaaaaaaaazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzztttttttttttttttttttttttttttttttt \n \n";
		//}


		//cv::Mat obj_img = left;// (img_size, CV_8UC1);
		//
		//for (auto & obj : objects)
		//{
		//	for (int i = 1; i < obj.polyg.size(); ++i)
		//	{
		//		cv::line(obj_img, cv::Point(obj.polyg[i - 1].x(), obj.polyg[i - 1].y()), cv::Point(obj.polyg[i].x(), obj.polyg[i].y()),
		//			256, 3);
		//	}
		//}
		//
		//
		//for (int i = 1; i < p.polyg.size(); ++i)
		//{
		//	cv::line(obj_img, cv::Point2f(p.polyg[i-1].x(), p.polyg[i - 1].y()), cv::Point2f(p.polyg[i].x(), p.polyg[i].y()), 255, 5);
		//}





		//demo.swap_buffer();
		frame_no++;

		//for (int i = 0; i < img_size.height; ++i)
		//	cv::circle(v_disp, cv::Point2f(v_disp_road[i], i), 1, 128, 2);

		//for (int i = 0; i < height / 16; ++i)
		//{
		//	cv::circle(cu_disp, cv::Point(250, height - i * 16 - 1), 3, 255, 1);
		//	cv::putText(cu_disp, std::to_string(i), cv::Point(270, height - i * 16 - 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, 255);
		//	cv::line(cu_disp, cv::Point(300, height - i * 16 - 1), cv::Point(1200, height - i * 16 - 1), 10, 1);
		//}

		//for (int i = 0; i < width; ++i)
		//{
		//	cv::circle(u_disp, cv::Point(i, free_space[i]), 3, 255, 1);
		//}
		//cv::imshow("v_disp", u_disp * 1024);
		//
		//cv::imshow("free_space_voting_res", free_space_voting_res);
		//
		//cv::waitKey(1);

	}

	delete d_output_buffer;
}
