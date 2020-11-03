#include "sgm.hpp"
#include "census_transform.hpp"
#include "path_aggregation.h"
#include "winner_takes_all.hpp"

namespace sgm
{
namespace cl
{
template <typename input_type, size_t MAX_DISPARITY>
class SemiGlobalMatching<input_type, MAX_DISPARITY>::Impl {

private:
    CensusTransform<input_type> m_census;
    PathAggregation<MAX_DISPARITY> m_path_aggregation;
    WinnerTakesAll<MAX_DISPARITY> m_winner_takes_all;

public:
    Impl(cl_context ctx, cl_device_id device)
        : m_census(ctx, device)
        , m_path_aggregation(ctx, device)
        , m_winner_takes_all(ctx, device)
    { 
    }

    void enqueue(
        DeviceBuffer<output_type> & dest_left,
        DeviceBuffer<output_type> & dest_right,
        const DeviceBuffer<input_type> & src_left,
        const DeviceBuffer<input_type>& src_right,
        DeviceBuffer<feature_type> & feature_buffer_left,
        DeviceBuffer<feature_type> & feature_buffer_right,
        int width,
        int height,
        int src_pitch,
        int dst_pitch,
        const Parameters& param,
        cl_command_queue stream)
    {
        m_census.enqueue(
            src_left, feature_buffer_left, width, height, src_pitch, stream);
        m_census.enqueue(
            src_right, feature_buffer_right, width, height, src_pitch, stream);
        m_path_aggregation.enqueue(
            feature_buffer_left,
            feature_buffer_right,
            width, height,
            param.path_type,
            param.P1,
            param.P2,
            param.min_disp,
            stream);
        m_winner_takes_all.enqueue(
            dest_left, dest_right,
            m_path_aggregation.get_output(),
            width, height, dst_pitch,
            param.uniqueness, param.subpixel, param.path_type,
            stream);
    }

};



template<typename input_type, size_t MAX_DISPARITY>
inline sgm::cl::SemiGlobalMatching<input_type, MAX_DISPARITY>::SemiGlobalMatching(cl_context context, cl_device_id device)
    : m_impl(std::make_unique<Impl>(context, device))
{
}

template<typename input_type, size_t MAX_DISPARITY>
SemiGlobalMatching<input_type, MAX_DISPARITY>::~SemiGlobalMatching()
{
}

template<typename input_type, size_t MAX_DISPARITY>
void SemiGlobalMatching<input_type, MAX_DISPARITY>::enqueue(
    DeviceBuffer<output_type> & dest_left,
    DeviceBuffer<output_type> & dest_right,
    const DeviceBuffer<input_type> & src_left,
    const DeviceBuffer<input_type> & src_right,
    DeviceBuffer<feature_type>& feature_buffer_left,
    DeviceBuffer<feature_type>& feature_buffer_right,
    int width,
    int height,
    int src_pitch,
    int dst_pitch,
    const Parameters& param,
    cl_command_queue stream)
{
    m_impl->enqueue(
        dest_left,
        dest_right,
        src_left,
        src_right,
        feature_buffer_left,
        feature_buffer_right,
        width, height,
        src_pitch, dst_pitch,
        param,
        stream);
}



template class SemiGlobalMatching<uint8_t, 64>;
template class SemiGlobalMatching<uint8_t, 128>;
template class SemiGlobalMatching<uint8_t, 256>;
template class SemiGlobalMatching<uint16_t, 64>;
template class SemiGlobalMatching<uint16_t, 128>;
template class SemiGlobalMatching<uint16_t, 256>;

}
}