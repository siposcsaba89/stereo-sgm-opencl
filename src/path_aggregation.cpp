#include "path_aggregation.h"
//debugging
#include <opencv2/opencv.hpp>
#include <fstream>
namespace sgm 
{
namespace cl
{
template<size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::PathAggregation(cl_context ctx, cl_device_id device, PathType path_type, int width, int height)
    : m_cost_buffer(ctx)
    , m_down2up(ctx, device)
    , m_up2down(ctx, device)
    , m_right2left(ctx, device)
    , m_left2right(ctx, device)
    , m_upleft2downright(ctx, device)
    , m_upright2downleft(ctx, device)
    , m_downright2upleft(ctx, device)
    , m_downleft2upright(ctx, device)
    , m_path_type(path_type)
    , m_width(width)
    , m_height(height)
{
    for (size_t i = 0; i < MAX_NUM_PATHS; ++i)
    {
        cl_int err;
        m_streams[i] = clCreateCommandQueue(ctx, device, 0, &err);
        CHECK_OCL_ERROR(err, "Failed to create command queue");
    }

    //allocating memory
    const unsigned int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;
    const size_t buffer_size = width * height * MAX_DISPARITY * num_paths;
    const size_t buffer_step = width * height * MAX_DISPARITY;
    if (m_cost_buffer.size() != buffer_size)
    {
        m_cost_buffer.allocate(buffer_size);
        m_sub_buffers.resize(num_paths);
        for (unsigned i = 0; i < num_paths; ++i)
        {
            cl_buffer_region region = { buffer_step * i, buffer_step };
            cl_int err;
            m_sub_buffers[i].setBufferData(nullptr, buffer_step, clCreateSubBuffer(m_cost_buffer.data(),
                CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region, &err));
            CHECK_OCL_ERROR(err, "Error creating subbuffer!");
        }
    }
}

template<size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::~PathAggregation()
{
    for (int i = 0; i < MAX_NUM_PATHS; ++i)
    {
        if (m_streams[i])
        {
            clReleaseCommandQueue(m_streams[i]);
            m_streams[i] = nullptr;
        }
    }
}

template<size_t MAX_DISPARITY>
const DeviceBuffer<cost_type>& PathAggregation<MAX_DISPARITY>::get_output() const
{
    return m_cost_buffer;
}

template<size_t MAX_DISPARITY>
void PathAggregation<MAX_DISPARITY>::enqueue(const DeviceBuffer<feature_type>& left,
    const DeviceBuffer<feature_type>& right,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    cl_command_queue stream)
{
    cl_int err = clFinish(stream);
    CHECK_OCL_ERROR(err, "Error finishing queue");
    m_up2down.enqueue(
        m_sub_buffers[0],
        left,
        right,
        m_width,
        m_height,
        p1,
        p2,
        min_disp,
        m_streams[0]
    );
    m_down2up.enqueue(
        m_sub_buffers[1],
        left,
        right,
        m_width,
        m_height,
        p1,
        p2,
        min_disp,
        m_streams[1]
    );
    m_left2right.enqueue(
        m_sub_buffers[2],
        left,
        right,
        m_width,
        m_height,
        p1,
        p2,
        min_disp,
        m_streams[2]
    );
    m_right2left.enqueue(
        m_sub_buffers[3],
        left,
        right,
        m_width,
        m_height,
        p1,
        p2,
        min_disp,
        m_streams[3]
    );

    if (m_path_type == PathType::SCAN_8PATH)
    {

        //{
        //    clFinish(stream);
        //    cv::Mat cost(height, width, CV_8UC4);
        //    clEnqueueReadBuffer(stream, right.data(), true, 0, width * height * 4, cost.data, 0, nullptr, nullptr);
        //    cv::imwrite("debug_cost_ocl.tiff", cost);
        //    cv::imshow("cost_res_ocl", cost);
        //    cv::waitKey(0);
        //}


        m_upleft2downright.enqueue(
            m_sub_buffers[4],
            left,
            right,
            m_width,
            m_height,
            p1,
            p2,
            min_disp,
            m_streams[4]
        );
        //{
        //    int path_id = 4;
        //    clFinish(m_streams[path_id]);
        //
        //    cl_buffer_region region = { buffer_step * path_id + width * height * 0, width * height };
        //    cl_mem buff = clCreateSubBuffer(m_cost_buffer.data(),
        //        CL_MEM_READ_WRITE,
        //        CL_BUFFER_CREATE_TYPE_REGION,
        //        &region, &err);
        //
        //
        //    cv::Mat cost(height, width, CV_8UC1);
        //    clEnqueueReadBuffer(stream, buff, true, 0, width * height * 1, cost.data, 0, nullptr, nullptr);
        //    {
        //        std::ofstream mat_txt("debug_cost_ocl.txt");
        //        for (int j = 0; j < cost.rows; ++j)
        //        {
        //            for (int i = 0; i < cost.cols; ++i)
        //            {
        //                mat_txt << static_cast<int>(cost.at<uint8_t>(j, i)) << " ";
        //            }
        //            mat_txt << std::endl;
        //        }
        //        mat_txt.close();
        //    }
        //    //cv::imwrite("debug_cost_ocl.tiff", cost);
        //    cv::imshow("cost_res_ocl", cost);
        //    cv::waitKey(0);
        //}

        m_upright2downleft.enqueue(
            m_sub_buffers[5],
            left,
            right,
            m_width,
            m_height,
            p1,
            p2,
            min_disp,
            m_streams[5]
        );
        m_downright2upleft.enqueue(
            m_sub_buffers[6],
            left,
            right,
            m_width,
            m_height,
            p1,
            p2,
            min_disp,
            m_streams[6]
        );

        m_downleft2upright.enqueue(
            m_sub_buffers[7],
            left,
            right,
            m_width,
            m_height,
            p1,
            p2,
            min_disp,
            m_streams[7]
        );
    }

    const unsigned int num_paths = m_path_type == PathType::SCAN_4PATH ? 4 : 8;
    for (unsigned i = 0; i < num_paths; ++i)
    {
        cl_int err = clFinish(m_streams[i]);
        CHECK_OCL_ERROR(err, "Error finishing path aggregation!");
    }
}

template class PathAggregation<64>;
template class PathAggregation<128>;
template class PathAggregation<256>;


template<int DIRECTION, unsigned int MAX_DISPARITY>
inline void VerticalPathAggregation<DIRECTION, MAX_DISPARITY>::init()
{
    if (m_kernel == nullptr)
    {
        //reading cl files
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_utility = fs.open("src/ocl/utility.cl");
        auto kernel_path_aggregation_common = fs.open("src/ocl/path_aggregation_common.cl");
        auto kernel_path_aggregation_vertical = fs.open("src/ocl/path_aggregation_vertical.cl");

        std::string kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_utility.begin(), kernel_utility.end())
            + std::string(kernel_path_aggregation_common.begin(), kernel_path_aggregation_common.end())
            + std::string(kernel_path_aggregation_vertical.begin(), kernel_path_aggregation_vertical.end());

        //Vertical path aggregation templates
        std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
        std::string kernel_direction = "#define DIRECTION " + std::to_string(DIRECTION) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
        kernel_src = std::regex_replace(kernel_src, std::regex("@DIRECTION@"), kernel_direction);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        //@DP_BLOCK_SIZE@
        //@SUBGROUP_SIZE@
        //@SIZE@
        //@BLOCK_SIZE@
         //Vertical path aggregation templates
        std::string kernel_DP_BLOCK_SIZE = "#define DP_BLOCK_SIZE " + std::to_string(DP_BLOCK_SIZE) + "\n";
        std::string kernel_SUBGROUP_SIZE = "#define SUBGROUP_SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
        std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

        //std::cout << "vertical_path_aggregation combined: " << std::endl;
        //std::cout << kernel_src << std::endl;

        m_program.init(m_cl_ctx, m_cl_device, kernel_src);
        //DEBUG
        m_kernel = m_program.getKernel("aggregate_vertical_path_kernel");
    }
}
//down2up
template struct VerticalPathAggregation<-1, 64>;
template struct VerticalPathAggregation<-1, 128>;
template struct VerticalPathAggregation<-1, 256>;
//up2down
template struct VerticalPathAggregation<1, 64>;
template struct VerticalPathAggregation<1, 128>;
template struct VerticalPathAggregation<1, 256>;


template<int DIRECTION, unsigned int MAX_DISPARITY>
inline void HorizontalPathAggregation<DIRECTION, MAX_DISPARITY>::init()
{
    if (m_kernel == nullptr)
    {
        //reading cl files
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_utility = fs.open("src/ocl/utility.cl");
        auto kernel_path_aggregation_common = fs.open("src/ocl/path_aggregation_common.cl");
        auto kernel_path_aggregation_horizontal = fs.open("src/ocl/path_aggregation_horizontal.cl");

        std::string kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_utility.begin(), kernel_utility.end())
            + std::string(kernel_path_aggregation_common.begin(), kernel_path_aggregation_common.end())
            + std::string(kernel_path_aggregation_horizontal.begin(), kernel_path_aggregation_horizontal.end());

        //Vertical path aggregation templates
        std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
        std::string kernel_direction = "#define DIRECTION " + std::to_string(DIRECTION) + "\n";
        std::string kernel_DP_BLOCKS_PER_THREAD = "#define DP_BLOCKS_PER_THREAD " + std::to_string(DP_BLOCKS_PER_THREAD) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
        kernel_src = std::regex_replace(kernel_src, std::regex("@DIRECTION@"), kernel_direction);
        kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCKS_PER_THREAD@"), kernel_DP_BLOCKS_PER_THREAD);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        //@DP_BLOCK_SIZE@
        //@SUBGROUP_SIZE@
        //@SIZE@
        //@BLOCK_SIZE@
         //path aggregation common templates
        std::string kernel_DP_BLOCK_SIZE = "#define DP_BLOCK_SIZE " + std::to_string(DP_BLOCK_SIZE) + "\n";
        std::string kernel_SUBGROUP_SIZE = "#define SUBGROUP_SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
        std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

        //std::cout << "horizontal_path_aggregation combined: " << std::endl;
        //std::cout << kernel_src << std::endl;

        m_program.init(m_cl_ctx, m_cl_device, kernel_src);
        //DEBUG
        m_kernel = m_program.getKernel("aggregate_horizontal_path_kernel");
    }
}
//down2up
template struct HorizontalPathAggregation<-1, 64>;
template struct HorizontalPathAggregation<-1, 128>;
template struct HorizontalPathAggregation<-1, 256>;
//up2down
template struct HorizontalPathAggregation<1, 64>;
template struct HorizontalPathAggregation<1, 128>;
template struct HorizontalPathAggregation<1, 256>;

template<int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
void sgm::cl::ObliquePathAggregation<X_DIRECTION, Y_DIRECTION, MAX_DISPARITY>::init()
{
    if (m_kernel == nullptr)
    {
        //reading cl files
        auto fs = cmrc::ocl_sgm::get_filesystem();
        auto kernel_inttypes = fs.open("src/ocl/inttypes.cl");
        auto kernel_utility = fs.open("src/ocl/utility.cl");
        auto kernel_path_aggregation_common = fs.open("src/ocl/path_aggregation_common.cl");
        auto kernel_path_aggregation_horizontal = fs.open("src/ocl/path_aggregation_oblique.cl");

        std::string kernel_src = std::string(kernel_inttypes.begin(), kernel_inttypes.end())
            + std::string(kernel_utility.begin(), kernel_utility.end())
            + std::string(kernel_path_aggregation_common.begin(), kernel_path_aggregation_common.end())
            + std::string(kernel_path_aggregation_horizontal.begin(), kernel_path_aggregation_horizontal.end());

        //Vertical path aggregation templates
        std::string kernel_max_disparoty = "#define MAX_DISPARITY " + std::to_string(MAX_DISPARITY) + "\n";
        std::string kernel_x_direction = "#define X_DIRECTION " + std::to_string(X_DIRECTION) + "\n";
        std::string kernel_y_direction = "#define Y_DIRECTION " + std::to_string(Y_DIRECTION) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@MAX_DISPARITY@"), kernel_max_disparoty);
        kernel_src = std::regex_replace(kernel_src, std::regex("@X_DIRECTION@"), kernel_x_direction);
        kernel_src = std::regex_replace(kernel_src, std::regex("@Y_DIRECTION@"), kernel_y_direction);

        static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
        //@DP_BLOCK_SIZE@
        //@SUBGROUP_SIZE@
        //@SIZE@
        //@BLOCK_SIZE@
         //path aggregation common templates
        std::string kernel_DP_BLOCK_SIZE = "#define DP_BLOCK_SIZE " + std::to_string(DP_BLOCK_SIZE) + "\n";
        std::string kernel_SUBGROUP_SIZE = "#define SUBGROUP_SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        std::string kernel_BLOCK_SIZE = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
        std::string kernel_SIZE = "#define SIZE " + std::to_string(SUBGROUP_SIZE) + "\n";
        kernel_src = std::regex_replace(kernel_src, std::regex("@DP_BLOCK_SIZE@"), kernel_DP_BLOCK_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SUBGROUP_SIZE@"), kernel_SUBGROUP_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@SIZE@"), kernel_SIZE);
        kernel_src = std::regex_replace(kernel_src, std::regex("@BLOCK_SIZE@"), kernel_BLOCK_SIZE);

        //std::cout << "horizontal_path_aggregation combined: " << std::endl;
        //std::cout << kernel_src << std::endl;

        m_program.init(m_cl_ctx, m_cl_device, kernel_src);
        //DEBUG
        m_kernel = m_program.getKernel("aggregate_oblique_path_kernel");
    }
}


//upleft2downright
template struct ObliquePathAggregation<1, 1, 64>;
template struct ObliquePathAggregation<1, 1, 128>;
template struct ObliquePathAggregation<1, 1, 256>;
//upright2downleft
template struct ObliquePathAggregation<-1, 1, 64>;
template struct ObliquePathAggregation<-1, 1, 128>;
template struct ObliquePathAggregation<-1, 1, 256>;
//downright2upleft
template struct ObliquePathAggregation<-1, -1, 64>;
template struct ObliquePathAggregation<-1, -1, 128>;
template struct ObliquePathAggregation<-1, -1, 256>;
//downleft2upright
template struct ObliquePathAggregation<1, -1, 64>;
template struct ObliquePathAggregation<1, -1, 128>;
template struct ObliquePathAggregation<1, -1, 256>;

}
}
