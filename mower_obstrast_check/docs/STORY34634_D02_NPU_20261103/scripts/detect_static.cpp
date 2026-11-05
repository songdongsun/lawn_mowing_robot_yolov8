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
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
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
// USB capture adapted from the existing standalone main_usb.cc.
static int camera_ioctl(int fd, unsigned long request, void* arg) {
    int result;
    do { result = ioctl(fd, request, arg); } while (result == -1 && errno == EINTR);
    return result;
}

class MjpegCamera {
public:
    ~MjpegCamera() { close(); }

    bool open(const std::string& device, int width, int height, int fps) {
        fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) { perror("open camera"); return false; }
        v4l2_capability capability{};
        if (camera_ioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            perror("VIDIOC_QUERYCAP"); return false;
        }
        const unsigned caps = capability.capabilities & V4L2_CAP_DEVICE_CAPS
                                  ? capability.device_caps : capability.capabilities;
        if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING)) {
            fprintf(stderr, "Camera does not support V4L2 streaming capture\n");
            return false;
        }
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = width;
        format.fmt.pix.height = height;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        if (camera_ioctl(fd_, VIDIOC_S_FMT, &format) < 0 ||
            format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
            fprintf(stderr, "Camera could not select MJPEG format\n");
            return false;
        }
        printf("Camera %s: %ux%u MJPEG\n", device.c_str(),
               format.fmt.pix.width, format.fmt.pix.height);
        v4l2_streamparm rate{};
        rate.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        rate.parm.capture.timeperframe.numerator = 1;
        rate.parm.capture.timeperframe.denominator = fps;
        if (camera_ioctl(fd_, VIDIOC_S_PARM, &rate) < 0)
            fprintf(stderr, "Camera did not accept requested FPS; continuing\n");
        v4l2_requestbuffers request{};
        request.count = 4;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        if (camera_ioctl(fd_, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
            perror("VIDIOC_REQBUFS"); return false;
        }
        for (unsigned i = 0; i < request.count; ++i) {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = i;
            if (camera_ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
                perror("VIDIOC_QUERYBUF"); return false;
            }
            void* data = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                              MAP_SHARED, fd_, buffer.m.offset);
            if (data == MAP_FAILED) { perror("mmap camera"); return false; }
            buffers_.push_back({data, buffer.length});
            if (camera_ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
                perror("VIDIOC_QBUF"); return false;
            }
        }
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (camera_ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            perror("VIDIOC_STREAMON"); return false;
        }
        streaming_ = true;
        return true;
    }

    bool read(cv::Mat& bgr, double& timestamp_ms, unsigned& sequence, bool& monotonic) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        timeval timeout{2, 0};
        const int ready = select(fd_ + 1, &fds, nullptr, nullptr, &timeout);
        if (ready <= 0) return false;
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (camera_ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) return false;
        // Drain ready buffers so inference consumes the latest frame, not a growing queue.
        for (size_t i=1;i<buffers_.size();++i) {
            v4l2_buffer newer{};
            newer.type=buffer.type;newer.memory=buffer.memory;
            if (camera_ioctl(fd_,VIDIOC_DQBUF,&newer)<0) {
                if (errno==EAGAIN) break;
                return false;
            }
            if (camera_ioctl(fd_,VIDIOC_QBUF,&buffer)<0) return false;
            buffer=newer;
        }
        timestamp_ms=buffer.timestamp.tv_sec*1000.0+buffer.timestamp.tv_usec/1000.0;
        sequence=buffer.sequence;
        monotonic=(buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK)==V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
        if (buffer.index < buffers_.size() && buffer.bytesused > 0 &&
            buffer.bytesused <= buffers_[buffer.index].length) {
            cv::Mat jpeg(1, buffer.bytesused, CV_8UC1, buffers_[buffer.index].data);
            try { bgr = cv::imdecode(jpeg, cv::IMREAD_COLOR); }
            catch (const cv::Exception& error) {
                fprintf(stderr, "JPEG decode failed: %s\n", error.what());
            }
        }
        if (camera_ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
            perror("VIDIOC_QBUF"); return false;
        }
        return !bgr.empty();
    }

    void close() {
        if (fd_ < 0) return;
        if (streaming_) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            camera_ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (const auto& buffer : buffers_) munmap(buffer.data, buffer.length);
        buffers_.clear();
        ::close(fd_);
        fd_ = -1;
    }

private:
    struct Buffer { void* data; size_t length; };
    int fd_ = -1;
    bool streaming_ = false;
    std::vector<Buffer> buffers_;
};

static volatile sig_atomic_t stopped=0;
static void stop_camera(int) { stopped=1; }
int main(int argc, char** argv) {
    try {
        using Clock=std::chrono::steady_clock;
        const bool live=argc==4 && std::string(argv[1])=="--camera";
        if (argc!=1 && !live) throw std::runtime_error("Usage: detectStatic [--camera /dev/video0 WEB_DIR]");
        std::signal(SIGINT,stop_camera);std::signal(SIGTERM,stop_camera);
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
        if (live && elements!=12*8400) throw std::runtime_error("Camera mode requires selected 8-class model");
        std::vector<float> raw(elements);
        vknn_output output{};
        output.user_allocated=false;output.original_tensor=true;
        std::ifstream list;
        std::ofstream timing;
        std::unique_ptr<MjpegCamera> camera;
        if (live) {
            camera=std::make_unique<MjpegCamera>();
            if (!camera->open(argv[2],640,480,30)) throw std::runtime_error("Camera open failed");
        } else {
            list.open("input_list.txt");timing.open("outputs/timing.csv");
            if (!list || !timing) throw std::runtime_error("Cannot open list/output");
            timing<<"id,inference_ms,postprocess_ms,boxes\n";
        }
        std::string id;int w,h,left,top,count=0;float gain;
        auto stream_start=Clock::now();double inference_sum=0;
        while (!stopped && (live || bool(list>>id>>w>>h>>gain>>left>>top))) {
            auto frame_start=Clock::now();
            cv::Mat original,image;double stamp=0;unsigned sequence=0;bool monotonic=false;
            if (live) {
                if (!camera->read(original,stamp,sequence,monotonic)) {
                    if (stopped) break;
                    throw std::runtime_error("Camera timeout or capture/decode failure");
                }
                w=original.cols;h=original.rows;gain=std::min(640.f/w,640.f/h);
                int nw=std::lround(w*gain),nh=std::lround(h*gain);
                left=std::lround((640-nw)/2.0-.1);top=std::lround((640-nh)/2.0-.1);
                cv::Mat resized;cv::resize(original,resized,{nw,nh},0,0,cv::INTER_LINEAR);
                cv::copyMakeBorder(resized,image,top,640-nh-top,left,640-nw-left,cv::BORDER_CONSTANT,{114,114,114});
            } else {
                if (id.find_first_not_of("0123456789")!=std::string::npos || w<=0 || h<=0 || gain<=0)
                    throw std::runtime_error("Invalid input metadata");
                image=cv::imread("inputs/"+id+".png");
            }
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
            if (live) {
                ++count;
                const double inference_ms=std::chrono::duration<double,std::milli>(b-a).count();
                inference_sum+=inference_ms;
                if (count%3==0) {
                    static const char* names[]={"person","cat","dog","frisbee","sports ball","teddy bear","garden hose","rock"};
                    for (const auto& box:boxes) {
                        cv::rectangle(original,{int(box.x1),int(box.y1)},{int(box.x2),int(box.y2)},{0,220,255},2);
                        char label[96];std::snprintf(label,sizeof(label),"%s %.2f",names[box.cls],box.score);
                        cv::Point pos{int(box.x1),std::max(18,int(box.y1)-5)};
                        cv::putText(original,label,pos,cv::FONT_HERSHEY_SIMPLEX,.5,{0,0,0},3);
                        cv::putText(original,label,pos,cv::FONT_HERSHEY_SIMPLEX,.5,{255,255,255},1);
                    }
                    const double fps=count/std::chrono::duration<double>(c-stream_start).count();
                    char label[128];std::snprintf(label,sizeof(label),"D02 KL | %.1f FPS | NPU %.1f ms | frame %d",fps,inference_ms,count);
                    cv::putText(original,label,{8,h-12},cv::FONT_HERSHEY_SIMPLEX,.5,{0,0,0},3);
                    cv::putText(original,label,{8,h-12},cv::FONT_HERSHEY_SIMPLEX,.5,{255,255,255},1);
                    const std::string dir=argv[3];
                    if (!cv::imwrite(dir+"/latest.tmp.jpg",original,{cv::IMWRITE_JPEG_QUALITY,85}) ||
                        std::rename((dir+"/latest.tmp.jpg").c_str(),(dir+"/latest.jpg").c_str()))
                        throw std::runtime_error("Cannot publish preview");
                    timespec now{};clock_gettime(CLOCK_MONOTONIC,&now);
                    const double age=monotonic ? now.tv_sec*1000.0+now.tv_nsec/1e6-stamp : -1;
                    std::ofstream json(dir+"/latest.tmp.json");json<<std::setprecision(12);
                    json<<"{\"frame\":"<<count<<",\"sequence\":"<<sequence<<",\"capture_timestamp_ms\":"<<stamp
                        <<",\"capture_clock_monotonic\":"<<(monotonic?"true":"false")<<",\"frame_age_ms\":"<<age
                        <<",\"fps\":"<<fps<<",\"inference_ms\":"<<inference_ms<<",\"mean_inference_ms\":"<<inference_sum/count
                        <<",\"processing_ms\":"<<std::chrono::duration<double,std::milli>(Clock::now()-frame_start).count()<<",\"boxes\":[";
                    for (size_t i=0;i<boxes.size();++i) {const auto& v=boxes[i];if(i)json<<",";
                        json<<"["<<v.x1<<","<<v.y1<<","<<v.x2<<","<<v.y2<<","<<v.score<<","<<v.cls<<"]";}
                    json<<"]}";json.close();
                    if (!json || std::rename((dir+"/latest.tmp.json").c_str(),(dir+"/latest.json").c_str()))
                        throw std::runtime_error("Cannot publish metadata");
                }
                if (count%300==0) std::cout<<"LIVE frames="<<count<<" mean_npu_ms="<<inference_sum/count<<std::endl;
                continue;
            }
            std::ofstream binary("outputs/"+id+".f32",std::ios::binary);
            binary.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(float));
            std::ofstream txt("outputs/"+id+".txt");txt<<std::setprecision(9);
            for (const auto& box:boxes) txt<<box.x1<<" "<<box.y1<<" "<<box.x2<<" "<<box.y2<<" "<<box.score<<" "<<box.cls<<"\n";
            if (!binary || !txt) throw std::runtime_error("Output write failed");
            timing<<id<<","<<std::chrono::duration<double,std::milli>(b-a).count()<<","<<std::chrono::duration<double,std::milli>(c-b).count()<<","<<boxes.size()<<"\n";
            std::cout<<"OK "<<id<<" boxes="<<boxes.size()<<std::endl;++count;
        }
        if (!live && (!list.eof() || count!=140)) throw std::runtime_error("Incomplete test list");
        std::cout<<"COMPLETE "<<count<<std::endl;
    } catch (const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
