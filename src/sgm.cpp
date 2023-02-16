#include "sgm.hpp"
#include "census_transform.hpp"
#include "path_aggregation.h"
#include "winner_takes_all.hpp"

namespace sgm
{
namespace cl
{
class SemiGlobalMatching::Impl
{
  private:
    Parameters m_param;
    CensusTransform m_census;
    PathAggregation m_path_aggregation;
    WinnerTakesAll m_winner_takes_all;
    int m_width;
    int m_height;
    int m_src_pitch;
    int m_dst_pitch;

  public:
    Impl(cl_context ctx,
        cl_device_id device,
        MaxDisparity max_disparity,
        int width,
        int height,
        int src_pitch,
        int dst_pitch,
        int input_bits,
        const Parameters& param)
        : m_param(param)
        , m_census(ctx, device, input_bits)
        , m_path_aggregation(ctx, device, param.path_type, max_disparity, width, height)
        , m_winner_takes_all(
              ctx, device, max_disparity, param.path_type, param.subpixel, dst_pitch, height)
        , m_width(width)
        , m_height(height)
        , m_src_pitch(src_pitch)
        , m_dst_pitch(dst_pitch)
    {
    }

    void enqueue(DeviceBuffer<uint16_t>& dest_left,
        DeviceBuffer<uint16_t>& dest_right,
        const cl_mem src_left,
        const cl_mem src_right,
        DeviceBuffer<uint32_t>& feature_buffer_left,
        DeviceBuffer<uint32_t>& feature_buffer_right,
        cl_command_queue stream)
    {
        m_census.enqueue(src_left, feature_buffer_left, m_width, m_height, m_src_pitch, stream);
        m_census.enqueue(src_right, feature_buffer_right, m_width, m_height, m_src_pitch, stream);
        m_path_aggregation.enqueue(feature_buffer_left,
            feature_buffer_right,
            m_param.P1,
            m_param.P2,
            m_param.min_disp,
            stream);
        m_winner_takes_all.enqueue(dest_left,
            dest_right,
            m_path_aggregation.get_output(),
            m_width,
            m_height,
            m_dst_pitch,
            m_param.uniqueness,
            stream);
    }
};

sgm::cl::SemiGlobalMatching::SemiGlobalMatching(cl_context context,
    cl_device_id device,
    MaxDisparity max_disparity,
    int width,
    int height,
    int src_pitch,
    int dst_pitch,
    int input_bits,
    const Parameters& param)
    : m_impl(std::make_unique<Impl>(
          context, device, max_disparity, width, height, src_pitch, dst_pitch, input_bits, param))
{
}

SemiGlobalMatching::~SemiGlobalMatching()
{
}

void SemiGlobalMatching::enqueue(DeviceBuffer<uint16_t>& dest_left,
    DeviceBuffer<uint16_t>& dest_right,
    const cl_mem src_left,
    const cl_mem src_right,
    DeviceBuffer<uint32_t>& feature_buffer_left,
    DeviceBuffer<uint32_t>& feature_buffer_right,
    cl_command_queue stream)
{
    m_impl->enqueue(dest_left,
        dest_right,
        src_left,
        src_right,
        feature_buffer_left,
        feature_buffer_right,
        stream);
}

} // namespace cl
} // namespace sgm
