#include "tracking.hpp"
#include <iostream>

static void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
    std::cout << "PASS " << message << std::endl;
}
static Object det(int cls, float score, float x) { return {{x, 100, 50, 100}, cls, score}; }
static const TrackResult& get(const std::vector<TrackResult>& rows, int id) {
    for (const auto& t : rows) if (t.id == id) return t;
    throw std::runtime_error("Expected ID missing");
}
int main(int argc, char** argv) {
    try {
        auto config = TrackingConfig::load("tracker.conf");
        Tracking t(config);
        // Replay the exact detections/timestamps recorded from the real camera, without the NPU.
        if (argc == 2 || (argc == 3 && std::string(argv[2]) == "--features")) {
            std::ifstream input(argv[1]);
            int frame, count, previous_frame = -1; double stamp;
            if (!input) throw std::runtime_error("Cannot open replay");
            while (input >> frame >> stamp >> count) {
                if (frame <= previous_frame || count < 0 || count > 300) throw std::runtime_error("Invalid replay frame");
                previous_frame = frame;
                std::vector<Object> objects;
                for (int i = 0; i < count; ++i) {
                    Object o;
                    if (!(input >> o.label >> o.prob >> o.rect.x >> o.rect.y >> o.rect.width >> o.rect.height))
                        throw std::runtime_error("Incomplete replay detection");
                    if (argc==3) {
                        int dimensions;
                        if (!(input>>dimensions) || (dimensions!=0 && dimensions!=256)) throw std::runtime_error("Invalid feature dimensions");
                        o.feature.resize(dimensions);
                        for(float& value:o.feature) if(!(input>>value)) throw std::runtime_error("Incomplete descriptor");
                    }
                    objects.push_back(o);
                }
                std::cout << "{\"frame\":" << frame << ",\"tracks\":";
                write_tracks(std::cout, t.update(objects, stamp)); std::cout << "}\n";
            }
            if (!input.eof()) throw std::runtime_error("Invalid replay file");
            return 0;
        }
        require(t.update({det(0, .2, 100)}, 0).empty(), "low confidence cannot create a track");
        auto rows = t.update({det(0, .9, 100)}, 70);
        int id = rows.at(0).id;
        require(rows.size() == 1 && rows[0].state == "new", "new track requires confirmation");
        rows = t.update({det(0, .9, 102)}, 140);
        require(get(rows, id).state == "tracked", "second high detection confirms same ID");
        rows = t.update({det(0, .2, 104)}, 210);
        require(get(rows, id).state == "tracked" && get(rows, id).score < .25, "low confidence second association retains ID");
        rows = t.update({}, 280);
        require(get(rows, id).state == "lost" && !get(rows, id).observed, "empty frame produces unobserved lost state");
        rows = t.update({det(0, .9, 106)}, 700);
        require(get(rows, id).state == "tracked", "short occlusion recovers original ID");
        rows = t.update({det(1, .9, 106)}, 770);
        int cat = -1; for (const auto& row : rows) if (row.cls == 1) cat = row.id;
        require(cat != -1 && cat != id && get(rows, id).state == "lost", "overlapping different class cannot inherit ID");
        rows = t.update({det(1, .2, 106)}, 840);
        require(get(rows, cat).state == "removed", "unconfirmed track is not confirmed by low detection");
        rows = t.update({}, 1750);
        require(get(rows, id).state == "removed", "expiry uses elapsed milliseconds");
        require(t.update({}, 1820).empty(), "removed event emitted once");
        rows = t.update({det(0, .9, 106)}, 1890);
        int next = rows.at(0).id;
        require(next != id, "long occlusion starts new ID");
        rows = t.update({det(0, .9, 106)}, 3100);
        require(get(rows, next).state == "removed" && rows.size() == 2, "stream gap expires ID before association");
        auto ending = t.finish();
        require(ending.size() == 1 && ending[0].state == "removed" && !ending[0].observed, "stream shutdown removes active tracks");
        Tracking multi(config);
        rows = multi.update({det(4, .9, 30), det(4, .9, 300)}, 0);
        int first = rows[0].id, second = rows[1].id;
        rows = multi.update({det(4, .9, 298), det(4, .9, 32)}, 70);
        require(get(rows, first).xyxy[0] < 100 && get(rows, second).xyxy[0] > 200, "same-class association survives detection order change");
        for (int i = 2; i < 5000; ++i) rows = multi.update({}, i * 70.0);
        require(rows.empty(), "long empty sequence has no retained tracks or repeated removal events");
        Tracking appearance(config);
        auto left=det(0,.9,100),right=det(0,.9,120);
        left.feature.assign(256,0);right.feature.assign(256,0);left.feature[0]=1;right.feature[1]=1;
        rows=appearance.update({left,right},0);int aid=rows[0].id,bid=rows[1].id;
        appearance.update({left,right},70);
        left.rect.x=120;right.rect.x=100;
        rows=appearance.update({right,left},140);
        require(get(rows,aid).xyxy[0]>get(rows,bid).xyxy[0],"appearance preserves identities when overlapping people exchange positions");
        std::cout << "ALL_CHECKS_PASSED\n";
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
