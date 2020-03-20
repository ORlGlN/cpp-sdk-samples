#pragma once

#include <Core.h>
#include "ProgressBar.h"

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>

class VideoReader {
public:
    VideoReader(const boost::filesystem::path& file_path, unsigned int sampling_frame_rate);

    bool GetFrame(cv::Mat& bgr_frame, affdex::Timestamp& timestamp_ms);
    bool GetFrameData(cv::Mat& bgr_frame, affdex::Timestamp& timestamp_ms);

    uint64_t TotalFrames() const;

private:
    cv::VideoCapture cap;
    affdex::Timestamp last_timestamp_ms;
    const unsigned int sampling_frame_rate;

    uint64_t total_frames;
    uint64_t current_frame = 0;
    std::unique_ptr<ProgressBar> frame_progress;
};