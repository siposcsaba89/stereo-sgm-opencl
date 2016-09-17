#pragma once

#include <CL/cl.hpp>

class StereoSGM
{
public:
	StereoSGM();
	~StereoSGM();
private:
	void initCL();
private:
	cl::Context m_context;
	cl::CommandQueue m_command_queue;
	cl::Program m_program;
	cl::Device m_device;
	//cl::Kernel 
	//buffers




};