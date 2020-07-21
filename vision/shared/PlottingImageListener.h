#pragma once

#include "ImageListener.h"
#include "PlottingListener.h"

using namespace affdex;

class PlottingImageListener : public vision::ImageListener, public PlottingListener<vision::Face> {

public:

    PlottingImageListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_face_id) :
        PlottingListener(csv, draw_display, enable_logging), capture_last_ts_(0), process_fps_(0), capture_fps_(0),
        draw_face_id_(draw_face_id), frames_with_faces_(0) {
        out_stream_ << "TimeStamp,faceId,upperLeftX,upperLeftY,lowerRightX,lowerRightY,confidence,interocularDistance,";
        for (const auto& angle : viz_.HEAD_ANGLES) {
            out_stream_ << angle.second << ",";
        }
        for (const auto& emotion : viz_.EMOTIONS) {
            out_stream_ << emotion.second << ",";
        }
        for (const auto& expression : viz_.EXPRESSIONS) {
            out_stream_ << expression.second << ",";
        }
        out_stream_ << "mood,dominantEmotion,dominantEmotionConfidence,gaze, gazeConfidence";
        out_stream_ << "identity,identityConfidence,age,ageConfidence,ageCategory";
        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;
    }

    int getCaptureFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return capture_fps_;
    }

    void onImageResults(std::map<vision::FaceId, vision::Face> faces, vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        const int diff = image.getTimestamp() - process_last_ts_;
        if (diff > 0) {
            results_.emplace_back(image, faces);
            process_fps_ = 1000.0f / diff;
            process_last_ts_ = image.getTimestamp();

            processed_frames_++;
            if (!faces.empty()) {
                frames_with_faces_++;
            }
        }

    };

    void onImageCapture(vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        const int diff = image.getTimestamp() - capture_last_ts_;
        if (diff > 0) {
            capture_fps_ = 1000.0f / diff;
            capture_last_ts_ = image.getTimestamp();
        }
    };

    void outputToFile(const std::map<vision::FaceId, vision::Face>& faces, double time_stamp) override {
        if (faces.empty()) {
            out_stream_ << time_stamp
                        << ",nan,nan,nan,nan,nan,nan,nan,"; // face ID, bbox UL X, UL Y, BR X, BR Y, confidence, interocular distance
            for (const auto& angle : viz_.HEAD_ANGLES) {
                out_stream_ << "nan,";
            }
            for (const auto& emotion : viz_.EMOTIONS) {
                out_stream_ << "nan,";
            }
            for (const auto& expression : viz_.EXPRESSIONS) {
                out_stream_ << "nan,";
            }
            // mood, dominant emotion, dominant emotion confidence, gaze, gaze-confidence, identity, identity_confidence, age, age_confidence, age_category
            out_stream_<< "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan";
            out_stream_ << std::endl;
        }

        for (const auto& face_id_pair : faces) {
            vision::Face f = face_id_pair.second;
            std::vector<vision::Point> bbox(f.getBoundingBox());

            out_stream_ << time_stamp << ","
                        << f.getId() << ","
                        << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                        << "," << std::setprecision(4)
                        << f.getConfidence() << ","
                        << f.getMeasurements().at(vision::Measurement::INTEROCULAR_DISTANCE) << ",";

            auto measurements = f.getMeasurements();
            for (const auto& m : viz_.HEAD_ANGLES) {
                out_stream_ << measurements.at(m.first) << ",";
            }

            auto emotions = f.getEmotions();
            for (const auto& emo : viz_.EMOTIONS) {
                out_stream_ << emotions.at(emo.first) << ",";
            }

            auto expressions = f.getExpressions();
            for (const auto& exp : viz_.EXPRESSIONS) {
                out_stream_ << expressions.at(exp.first) << ",";
            }

            vision::Mood mood = f.getMood();
            out_stream_ << viz_.MOODS[mood] << ",";

            vision::DominantEmotionMetric dominant_emotion_metric = f.getDominantEmotion();
            out_stream_ << viz_.DOMINANT_EMOTIONS[dominant_emotion_metric.dominantEmotion] << ","
                        << dominant_emotion_metric.confidence << ",";

            vision::GazeMetric gaze_metric = f.getGazeMetric();
            out_stream_ << viz_.GAZE[gaze_metric.gaze] << "," << gaze_metric.confidence << ",";

            auto identity_metric = f.getIdentityMetric();
            std::string id_content;
            identity_metric.id == -1 ? id_content = "UNKNOWN" : id_content = std::to_string(identity_metric.id);
            out_stream_ << id_content << "," << identity_metric.confidence << ",";

            auto age_metric = f.getAgeMetric();
            std::string age_content;
            age_metric.years == -1 ? age_content = "UNKNOWN" : age_content = std::to_string(age_metric.years);
            out_stream_ << age_content << "," << age_metric.confidence << ",";

            auto age_category = f.getAgeCategory();
            out_stream_ << viz_.AGE_CATEGORIES.at(age_category);

            out_stream_ << std::endl;
        }
    }

    void draw(const std::map<vision::FaceId, vision::Face>& faces, const vision::Frame& image) override {
        const cv::Mat img = *(image.getImage());
        viz_.updateImage(img);

        for (const auto& face_id_pair : faces) {
            vision::Face f = face_id_pair.second;

            std::map<vision::FacePoint, vision::Point> points = f.getFacePoints();

            // Draw bounding box
            auto bbox = f.getBoundingBox();
            const float valence = f.getEmotions().at(vision::Emotion::VALENCE);
            viz_.drawBoundingBox(bbox, valence);

            // Draw Facial Landmarks Points
            viz_.drawPoints(f.getFacePoints());

            // Draw a face on screen
            viz_.drawFaceMetrics(f, bbox, draw_face_id_);
        }

        viz_.showImage();
        image_data_ = viz_.getImageData();
    }

    void processResults() override {
        while (getDataSize() > 0) {
            const std::pair<vision::Frame, std::map<vision::FaceId, vision::Face>> dataPoint = getData();
            vision::Frame frame = dataPoint.first;
            const std::map<vision::FaceId, vision::Face> faces = dataPoint.second;

            if (draw_display_) {
                draw(faces, frame);
            }

            outputToFile(faces, frame.getTimestamp());

            if (logging_enabled_) {
                std::cout << "timestamp: " << frame.getTimestamp()
                          << " cfps: " << getCaptureFrameRate()
                          << " pfps: " << getProcessingFrameRate()
                          << " faces: " << faces.size() << std::endl;
            }
        }
    }

    int getFramesWithFaces() const {
        return frames_with_faces_;
    }

    int getFramesWithFacesPercent() {
        return (static_cast<float>(frames_with_faces_) / processed_frames_) * 100;
    }

    int getProcessingFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return process_fps_;
    }

    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        capture_last_ts_ = 0;
        capture_fps_ = 0;
        process_last_ts_ = 0;
        process_fps_ = 0;
        start_ = std::chrono::system_clock::now();
        processed_frames_ = 0;
        frames_with_faces_ = 0;
        results_.clear();
    }

private:
    Timestamp capture_last_ts_;
    int process_fps_;
    int capture_fps_;
    bool draw_face_id_;
    int frames_with_faces_;
};
