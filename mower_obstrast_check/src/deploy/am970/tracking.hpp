#pragma once
#include "BYTETracker.h"
#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <stdexcept>

struct TrackingConfig {
    float low, high, new_track, match;
    double lost_ms;
    static TrackingConfig load(const std::string& path) {
        TrackingConfig c{};
        std::ifstream in(path);
        std::string extra;
        if (!(in >> c.low >> c.high >> c.new_track >> c.match >> c.lost_ms) || (in >> extra) ||
            !(0 < c.low && c.low < c.high && c.high <= c.new_track && c.new_track <= 1 &&
              0 < c.match && c.match <= 1 && 0 < c.lost_ms && c.lost_ms <= 10000))
            throw std::runtime_error("Invalid tracker config: low high new_track match lost_ms");
        return c;
    }
};

struct TrackResult {
    int id, cls;
    std::string state;
    bool observed;
    float score;
    std::vector<float> xyxy;
    double last_observed_ms;
};

class Tracking {
    TrackingConfig config;
    std::array<std::unique_ptr<BYTETracker>, 8> trackers;
    std::map<int, TrackResult> previous;
    double last_timestamp = -1;
public:
    explicit Tracking(TrackingConfig c) : config(c) {
        for (auto& t : trackers) t = std::make_unique<BYTETracker>(c.high, c.new_track, c.match, c.lost_ms);
    }
    std::vector<TrackResult> update(const std::vector<Object>& objects, double timestamp_ms) {
        if (!std::isfinite(timestamp_ms) || timestamp_ms < 0 || timestamp_ms <= last_timestamp)
            throw std::runtime_error("Tracker requires increasing monotonic timestamps");
        last_timestamp = timestamp_ms;
        std::array<std::vector<Object>, 8> groups;
        for (const auto& o : objects) {
            if (o.label < 0 || o.label >= 8 || !std::isfinite(o.prob) || !std::isfinite(o.rect.x) ||
                !std::isfinite(o.rect.y) || !std::isfinite(o.rect.width) || !std::isfinite(o.rect.height))
                throw std::runtime_error("Invalid tracker detection");
            if (!o.feature.empty()) {
                if (o.label!=0 || o.feature.size()!=256) throw std::runtime_error("Expected 256-D person descriptor");
                double norm=0;for(float x:o.feature) {if(!std::isfinite(x)) throw std::runtime_error("Nonfinite descriptor");norm+=x*x;}
                if(norm<1e-12) throw std::runtime_error("Zero descriptor");
            }
            if (o.prob >= config.low && o.rect.width > 0 && o.rect.height > 0) groups[o.label].push_back(o);
        }
        std::map<int, TrackResult> current;
        for (int cls = 0; cls < 8; ++cls) {
            auto& tracker = *trackers[cls];
            tracker.update(groups[cls], timestamp_ms);
            for (const auto& t : tracker.tracked())
                current.emplace(t.track_id, TrackResult{t.track_id, cls, t.is_activated ? "tracked" : "new",
                    true, t.score, t.tlbr, t.observed_ms});
            for (const auto& t : tracker.lost())
                current.emplace(t.track_id, TrackResult{t.track_id, cls, "lost", false, 0, t.tlbr, t.observed_ms});
        }
        std::vector<TrackResult> out;
        for (const auto& item : current) out.push_back(item.second);
        for (const auto& item : previous) if (!current.count(item.first)) {
            auto t = item.second; t.state = "removed"; t.observed = false; t.score = 0; out.push_back(t);
        }
        previous = std::move(current);
        return out;
    }
    std::vector<TrackResult> finish() {
        std::vector<TrackResult> out;
        for (auto& item : previous) {
            auto t = item.second; t.state = "removed"; t.observed = false; t.score = 0; out.push_back(t);
        }
        previous.clear();
        return out;
    }
};

inline void write_tracks(std::ostream& out, const std::vector<TrackResult>& tracks) {
    out << std::setprecision(12) << "[";
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i]; if (i) out << ",";
        out << "{\"track_id\":" << t.id << ",\"class_id\":" << t.cls << ",\"state\":\"" << t.state
            << "\",\"observed\":" << (t.observed ? "true" : "false") << ",\"confidence\":";
        if (t.observed) out << t.score; else out << "null";
        out << ",\"last_observed_ms\":" << t.last_observed_ms << ",\"xyxy\":[";
        for (size_t j = 0; j < 4; ++j) { if (j) out << ","; out << t.xyxy[j]; }
        out << "]}";
    }
    out << "]";
}
