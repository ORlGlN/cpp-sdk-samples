#pragma once

#include "OccupantListener.h"
#include "PlottingListener.h"

using namespace affdex;

class PlottingOccupantListener : public vision::OccupantListener, public PlottingListener<vision::Occupant> {

public:

    PlottingOccupantListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_occupant_id, const
    Duration callback_interval, std::vector<CabinRegion> cabin_regions) :
        PlottingListener(csv, draw_display, enable_logging), callback_interval_(callback_interval),
        cabin_regions_(std::move(cabin_regions)), draw_occupant_id_(draw_occupant_id) {
        out_stream_ << "TimeStamp, occupantId, confidence, regionId,  upperLeftX, upperLeftY, lowerRightX, lowerRightY";

        for (const auto& cr :cabin_regions_) {
            out_stream_ << "," << "Region " << cr.id;
        }

        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;
    }

    Duration getCallbackInterval() const override {
        return callback_interval_;
    }

    void onOccupantResults(const std::map<vision::OccupantId, vision::Occupant>& occupants,
                           vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results_.emplace_back(frame, occupants);
        process_last_ts_ = frame.getTimestamp();

        processed_frames_++;
        if (!occupants.empty()) {
            frames_with_occupants_++;
        }
    };

    void outputToFile(const std::map<vision::OccupantId, vision::Occupant>& occupants, double time_stamp) override {
        if (occupants.empty()) {
            // TimeStamp occupantId confidence regionId upperLeftX upperLeftY lowerRightX lowerRightY"
            out_stream_ << time_stamp << ",nan,nan,nan,nan,nan,nan,nan";
            for (const auto& cr :cabin_regions_) {
                out_stream_ << ",nan";
            }
            out_stream_ << std::endl;
        }

        for (const auto& occupant_id_pair : occupants) {
            vision::Occupant occup = occupant_id_pair.second;
            std::vector<vision::Point> bbox({occup.boundingBox.getTopLeft(), occup.boundingBox.getBottomRight()});

            out_stream_ << time_stamp << ","
                        << occupant_id_pair.first << ","
                        << occup.matchedSeat.matchConfidence << "," << occup.matchedSeat.cabinRegion.id << ","
                        << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                        << std::setprecision(4);

            for (const auto& cr :cabin_regions_) {
                if (cr.id == occup.matchedSeat.cabinRegion.id) {
                    out_stream_ << "," << occup.matchedSeat.matchConfidence;
                }
                else {
                    out_stream_ << "," << 0;
                }
            }

            out_stream_ << std::endl;
        }
    }

    void draw(const std::map<vision::OccupantId, vision::Occupant>& occupants, const vision::Frame& image) override {
        std::shared_ptr<unsigned char> img_data = image.getBGRByteArray();
        const cv::Mat img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3, img_data.get());
        viz_.updateImage(img);

        for (const auto& id_occupant_pair : occupants) {
            const auto occupant =  id_occupant_pair.second;
            viz_.drawOccupantMetrics(occupant);
            //add occupant region detected
            const auto id = occupant.matchedSeat.cabinRegion.id;
            if(std::find(occupant_regions_.begin(), occupant_regions_.end(), id) == occupant_regions_.end()) {
                occupant_regions_.emplace_back(id);
            }
        }

        viz_.showImage();
        image_data_ = viz_.getImageData();
    }

    void processResults() override {
        while (getDataSize() > 0) {
            const std::pair<vision::Frame, std::map<vision::OccupantId, vision::Occupant>> dataPoint = getData();
            vision::Frame frame = dataPoint.first;
            const std::map<vision::OccupantId, vision::Occupant> occupants = dataPoint.second;

            if (draw_display_) {
                draw(occupants, frame);
            }

            outputToFile(occupants, frame.getTimestamp());

            if (logging_enabled_) {
                std::cout << "timestamp: " << frame.getTimestamp()
                          << " occupants: " << occupants.size() << std::endl;
            }
        }
    }


    double getSamplesWithOccupantsPercent() {
        return (static_cast<double>(frames_with_occupants_) / processed_frames_) * 100;
    }

    std::string getOccupantRegionsDetected() const {
        std::string occupant_regions;
        for(int i = 0; i<occupant_regions_.size(); ++i){
            if(i>0){
                occupant_regions += ", ";
            }
            occupant_regions += std::to_string(occupant_regions_[i]);
        }
        return occupant_regions;
    }

    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        process_last_ts_ = 0;
        start_ = std::chrono::system_clock::now();
        processed_frames_ = 0;
        frames_with_occupants_ = 0;
        results_.clear();
    }

private:
    Duration callback_interval_;
    std::vector<CabinRegion> cabin_regions_;
    std::vector<int> occupant_regions_;
    bool draw_occupant_id_;
    unsigned int frames_with_occupants_;
};