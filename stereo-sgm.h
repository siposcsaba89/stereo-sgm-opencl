#pragma once

#include <CL/cl.hpp>

class StereoSGM
{
public:
	StereoSGM();
	~StereoSGM();
private:
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Program m_program;
	
	//cl::Kernel 
	//buffers




};