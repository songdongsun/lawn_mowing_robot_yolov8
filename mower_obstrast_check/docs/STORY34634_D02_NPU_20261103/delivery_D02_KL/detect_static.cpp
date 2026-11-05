// Standalone 8/80-class audit: prepared BGR640 -> VKNN -> original-image boxes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include "Vknn.h"

struct Context {
    vknn_context value;
    Context() {
        if (vknn_init(&value, const_cast<char*>("network_binary.nb"), 0, nullptr) != VKNN_SUCC)
            throw std::runtime_error("vknn_init failed");
    }
    ~Context() { vknn_release(value); }
};
void check(int status) {
    if (status != VKNN_SUCC) throw std::runtime_error("VKNN operation failed: " + std::to_string(status));
}
struct Box { float x1, y1, x2, y2, score; int cls; };
float iou(const Box& a, const Box& b) {
    float inter = std::max(0.f, std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) *
                  std::max(0.f, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    return inter / ((a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter+1e-7f);
}
std::vector<Box> decode(const std::vector<float>& raw, float gain, int left, int top, int w, int h) {
    std::vector<Box> candidates, kept;
    for (int j=0;j<8400;++j) {
        int cls=0;
        for (int c=1;c<int(raw.size()/8400)-4;++c) if (raw[(c+4)*8400+j]>raw[(cls+4)*8400+j]) cls=c;
        float score=raw[(cls+4)*8400+j];
        if (score<=.25f) continue;
        float x=raw[j],y=raw[8400+j],bw=raw[16800+j],bh=raw[25200+j];
        candidates.push_back({x-bw/2,y-bh/2,x+bw/2,y+bh/2,score,cls});
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Box& a,const Box& b){return a.score>b.score;});
    for (const auto& box:candidates) {
        bool suppressed=false;
        for (const auto& other:kept) if (box.cls==other.cls && iou(box,other)>.7f) {suppressed=true;break;}
        if (!suppressed) kept.push_back(box);
        if (kept.size()==300) break;
    }
    for (auto& b:kept) {
        b.x1=std::clamp((b.x1-left)/gain,0.f,float(w));
        b.x2=std::clamp((b.x2-left)/gain,0.f,float(w));
        b.y1=std::clamp((b.y1-top)/gain,0.f,float(h));
        b.y2=std::clamp((b.y2-top)/gain,0.f,float(h));
    }
    return kept;
}
int main() {
    try {
        using Clock=std::chrono::steady_clock;
        auto start=Clock::now();
        Context ctx;
        std::cout<<"INIT_MS "<<std::chrono::duration<double,std::milli>(Clock::now()-start).count()<<std::endl;
        int inputs=0,outputs=0;
        bool pre=false,dequant=false;
        check(vknn_query(ctx.value,VKNN_QUERY_IN_NUM,&inputs));
        check(vknn_query(ctx.value,VKNN_QUERY_OUT_NUM,&outputs));
        check(vknn_query(ctx.value,VKNN_QUERY_EXIST_PREPROCESS_NODE,&pre));
        check(vknn_query(ctx.value,VKNN_QUERY_EXIST_DEQUANT_NODE,&dequant));
        if (inputs!=1 || outputs!=1 || !pre || !dequant) throw std::runtime_error("Unexpected model IO contract");
        vknn_io_attr attr{};
        check(vknn_query(ctx.value,VKNN_QUERY_OUTPUT_ATTR,&attr));
        uint64_t elements=1;
        std::cout<<"OUTPUT_SHAPE";
        for (uint32_t i=0;i<attr.shapes[0].n_dim;++i) {
            elements*=attr.shapes[0].dim_size[i];std::cout<<" "<<attr.shapes[0].dim_size[i];
        }
        std::cout<<std::endl;
        if (elements!=12*8400 && elements!=84*8400) throw std::runtime_error("Unexpected output elements");
        std::vector<float> raw(elements);
        vknn_output output{};
        output.user_allocated=false;output.original_tensor=true;
        std::ifstream list("input_list.txt");
        std::ofstream timing("outputs/timing.csv");
        if (!list || !timing) throw std::runtime_error("Cannot open list/output");
        timing<<"id,inference_ms,postprocess_ms,boxes\n";
        std::string id;int w,h,left,top,count=0;float gain;
        while (list>>id>>w>>h>>gain>>left>>top) {
            if (id.find_first_not_of("0123456789")!=std::string::npos || w<=0 || h<=0 || gain<=0)
                throw std::runtime_error("Invalid input metadata");
            cv::Mat image=cv::imread("inputs/"+id+".png");
            if (image.type()!=CV_8UC3 || image.rows!=640 || image.cols!=640 || !image.isContinuous())
                throw std::runtime_error("Expected prepared contiguous BGR640 input");
            vknn_input input{};
            input.buffer=image.data;input.size=640*640*3;input.type=VKNN_TENSOR_UINT8;
            auto a=Clock::now();
            check(vknn_process_input(ctx.value,&input,1));
            check(vknn_inference(ctx.value,nullptr));
            check(vknn_process_output(ctx.value,&output,1));
            if (!output.buffer || output.size!=raw.size()) throw std::runtime_error("Invalid returned output buffer");
            auto values=static_cast<const float*>(output.buffer);
            std::copy(values,values+output.size,raw.begin());
            auto b=Clock::now();
            for (float x:raw) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite output");
            auto boxes=decode(raw,gain,left,top,w,h);
            auto c=Clock::now();
            std::ofstream binary("outputs/"+id+".f32",std::ios::binary);
            binary.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(float));
            std::ofstream txt("outputs/"+id+".txt");txt<<std::setprecision(9);
            for (const auto& box:boxes) txt<<box.x1<<" "<<box.y1<<" "<<box.x2<<" "<<box.y2<<" "<<box.score<<" "<<box.cls<<"\n";
            if (!binary || !txt) throw std::runtime_error("Output write failed");
            timing<<id<<","<<std::chrono::duration<double,std::milli>(b-a).count()<<","<<std::chrono::duration<double,std::milli>(c-b).count()<<","<<boxes.size()<<"\n";
            std::cout<<"OK "<<id<<" boxes="<<boxes.size()<<std::endl;++count;
        }
        if (!list.eof() || count!=140) throw std::runtime_error("Incomplete test list");
        std::cout<<"COMPLETE "<<count<<std::endl;
    } catch (const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
