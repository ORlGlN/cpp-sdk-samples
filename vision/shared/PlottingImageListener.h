#pragma once

#include "Visualizer.h"

#include <ImageListener.h>

#include <deque>
#include <mutex>
#include <fstream>
#include <condition_variable>

#include <iostream>



using namespace affdex;

class PlottingImageListener : public vision::ImageListener {

public:

    PlottingImageListener(std::ofstream &csv, bool draw_display, bool enable_logging, bool draw_face_id) :
        draw_display(draw_display),
        capture_last_ts(0),
        capture_fps(-1.0f),
        process_last_ts(0),
        process_fps(-1.0f),
        out_stream(csv),
        start(std::chrono::system_clock::now()),
        processed_frames(0),
        frames_with_faces(0),
        draw_face_id(draw_face_id),
        logging_enabled(enable_logging) {
        out_stream << "TimeStamp,faceId,confidence,location,interocularDistance,";
        for (const auto& angle : viz.HEAD_ANGLES) out_stream << angle.second << ",";
        for (const auto& emotion : viz.EMOTIONS) out_stream << emotion.second << ",";
        for (const auto& expression : viz.EXPRESSIONS) out_stream << expression.second << ",";
        out_stream << std::endl;
        out_stream.precision(4);
        out_stream << std::fixed;
    }

    double getProcessingFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return process_fps;
    }

    double getCaptureFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return capture_fps;
    }

    int getDataSize() {
        std::lock_guard<std::mutex> lg(mtx);
        return results.size();
    }

    unsigned int getProcessedFrames() {
        return processed_frames;
    }

    unsigned int getFramesWithFaces() {
        return frames_with_faces;
    }

    double getFramesWithFacesPercent() {
        return (static_cast<double>(frames_with_faces) / processed_frames) * 100;
    }

    std::pair<vision::Frame, std::map<vision::FaceId, vision::Face>> getData() {
        std::lock_guard<std::mutex> lg(mtx);
        std::pair<vision::Frame, std::map<vision::FaceId, vision::Face>> dpoint = results.front();
        results.pop_front();
        return dpoint;
    }

    void waitForResult() {
        std::unique_lock< std::mutex > lock(result_mtx);
        result_received.wait(lock, [&] { return getDataSize() > 0; });
    }

    void onImageResults(std::map<vision::FaceId, vision::Face> faces, vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        results.emplace_back(image, faces);
        process_fps = 1000.0f / (image.getTimestamp() - process_last_ts) ;
        process_last_ts = image.getTimestamp();

        processed_frames++;
        if (faces.size() > 0)
        {
            frames_with_faces++;
        }

        std::unique_lock< std::mutex > lock(result_mtx);
        result_received.notify_one();
    };

    void onImageCapture(vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        capture_fps = 1000.0f / (image.getTimestamp() - capture_last_ts);
        capture_last_ts = image.getTimestamp();
    };

    void outputToFile(const std::map<vision::FaceId, vision::Face> faces, const double timeStamp) {
        if (faces.empty()) {
            out_stream << timeStamp
                << ",nan,nan,nan,nan,"; // face ID, confidence, location, interocular distance
            for (const auto& angle : viz.HEAD_ANGLES) out_stream << "nan,";
            for (const auto& emotion : viz.EMOTIONS) out_stream << "nan,";
            for (const auto& expression : viz.EXPRESSIONS) out_stream << "nan,";
            out_stream << std::endl;
        }

        for (auto & face_id_pair : faces) {
            vision::Face f = face_id_pair.second;

            out_stream << timeStamp << ","
                << f.getId() << ","
                << f.getConfidence() << ","
                << viz.LOCATIONS[f.getOccupantLocation()] << ","
                << f.getMeasurements().at(vision::Measurement::INTEROCULAR_DISTANCE) << ",";

            auto measurements = f.getMeasurements();
            for (auto m : viz.HEAD_ANGLES) {
                out_stream << measurements.at(m.first) << ",";
            }

            auto emotions = f.getEmotions();
            for (auto emo : viz.EMOTIONS) {
                out_stream << emotions.at(emo.first) << ",";
            }

            auto expressions = f.getExpressions();
            for (auto exp : viz.EXPRESSIONS) {
                out_stream << expressions.at(exp.first) << ",";
            }

            out_stream << std::endl;
        }
    }

    void draw(const std::map<vision::FaceId, vision::Face> faces, const vision::Frame& image) {
        std::shared_ptr<unsigned char> imgdata = image.getBGRByteArray();
        const cv::Mat img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3, imgdata.get());
        viz.updateImage(img);

        for (auto & face_id_pair : faces) {
            vision::Face f = face_id_pair.second;
            
            std::map<vision::FacePoint, vision::Point> points = f.getFacePoints();
            
            // Draw Facial Landmarks Points
            viz.drawPoints(f.getFacePoints());

            // Draw bounding box
            auto bbox = f.getBoundingBox();
            const float valence = f.getEmotions().at(vision::Emotion::VALENCE);
            viz.drawBoundingBox(bbox, valence);

            // Draw a face on screen
            viz.drawFaceMetrics(f, bbox, draw_face_id);
        }

        viz.showImage();
    }

    void processResults() {
        while (getDataSize() > 0) {
            const std::pair<vision::Frame, std::map<vision::FaceId, vision::Face> > dataPoint = getData();
            vision::Frame frame = dataPoint.first;
            const std::map<vision::FaceId, vision::Face> faces = dataPoint.second;

            if (draw_display) {
                draw(faces, frame);
            }

            outputToFile(faces, frame.getTimestamp());

            if (logging_enabled) {
                std::cout << "timestamp: " << frame.getTimestamp()
                << " cfps: " << getCaptureFrameRate()
                << " pfps: " << getProcessingFrameRate()
                << " faces: "<< faces.size() << std::endl;
            }
        }
    }

    bool validate(std::set<vision::Expression> supported) {
        for (auto pair: viz.EXPRESSIONS) {
            if (supported.find(pair.first) == supported.end()) {
                std::cerr << "\"" << pair.second << "\": is not supported by the current data directory" << std::endl;
                return false;
            }
        }
        return true;
    }
    
    bool validate(std::set<vision::Emotion> supported) {
        for (auto pair: viz.EMOTIONS) {
            if (supported.find(pair.first) == supported.end()) {
                std::cerr << "\"" << pair.second << "\": is not supported by the current data directory" << std::endl;
                return false;
            }
        }
        return true;
    }

    bool validate(std::set<vision::Measurement> supported) {
        for (auto pair: viz.HEAD_ANGLES) {
            if (supported.find(pair.first) == supported.end()) {
                std::cerr << "\"" << pair.second << "\": is not supported by the current data directory" << std::endl;
                return false;
            }
        }
        return true;
    }

    std::map<vision::OccupantLocation, std::string> getLocationNames() const {
        return viz.LOCATIONS;
    }

private:
    bool draw_display;
    std::mutex mtx;
    std::mutex result_mtx;
    std::condition_variable result_received;
    std::deque<std::pair<vision::Frame, std::map<vision::FaceId, vision::Face> > > results;

    timestamp capture_last_ts;
    double capture_fps;
    timestamp process_last_ts;
    double process_fps;
    std::ofstream &out_stream;
    std::chrono::time_point<std::chrono::system_clock> start;

    Visualizer viz;

    unsigned int processed_frames;
    unsigned int frames_with_faces;
    bool draw_face_id;
    bool logging_enabled;
};
