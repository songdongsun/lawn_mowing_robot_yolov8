#include <Vknn.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kSide = 640;
constexpr int kAnchors = 8400;
constexpr int kChannels = 37;
constexpr int kProtos = 32;
constexpr int kProtoSide = 160;

struct Detection {
  int anchor;
  float score;
  cv::Rect2f box;
};

float iou(const cv::Rect2f &a, const cv::Rect2f &b) {
  const float intersection = (a & b).area();
  return intersection / (a.area() + b.area() - intersection);
}

size_t rss_kib() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      size_t value = 0;
      status >> value;
      return value;
    }
    std::getline(status, key);
  }
  return 0;
}
}  // namespace

class GrassSegNode : public rclcpp::Node {
 public:
  GrassSegNode() : Node("grass_seg_node") {
    const auto model = declare_parameter<std::string>("model_path", "");
    confidence_ = declare_parameter<double>("confidence", 0.25);
    nms_iou_ = declare_parameter<double>("nms_iou", 0.45);
    mask_threshold_ = declare_parameter<double>("mask_threshold", 0.5);
    stats_interval_ = declare_parameter<int>("stats_interval", 100);
    if (model.empty() || stats_interval_ < 1 || confidence_ < 0 || confidence_ > 1 ||
        nms_iou_ < 0 || nms_iou_ > 1 || mask_threshold_ < 0 || mask_threshold_ > 1) {
      throw std::invalid_argument("Invalid model path or threshold/statistics parameter");
    }
    if (vknn_init(&context_, const_cast<char *>(model.c_str()), 0, nullptr) != VKNN_SUCC) {
      throw std::runtime_error("vknn_init failed: " + model);
    }
    vknn_io_attr shape{};
    if (vknn_query(context_, VKNN_QUERY_OUTPUT_ATTR, &shape) != VKNN_SUCC || shape.num != 2 ||
        !matches(shape.shapes[0], {1, kChannels, kAnchors}) ||
        !matches(shape.shapes[1], {1, kProtos, kProtoSide, kProtoSide})) {
      vknn_release(context_);
      context_ = 0;
      throw std::runtime_error("Unexpected VKNN output shape; expected [1,37,8400], [1,32,160,160]");
    }
    prediction_.resize(kChannels * kAnchors);
    prototype_.resize(kProtos * kProtoSide * kProtoSide);
    auto qos = rclcpp::SensorDataQoS().keep_last(1);
    mask_pub_ = create_publisher<sensor_msgs::msg::Image>("mask", qos);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
          camera_width_ = msg->width;
          camera_height_ = msg->height;
        });
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "image", qos, [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(*msg); });
    window_start_ = std::chrono::steady_clock::now();
  }

  ~GrassSegNode() override {
    if (context_) vknn_release(context_);
  }

 private:
  static bool matches(const vknn_tensor_shape &shape, std::initializer_list<int> dims) {
    if (shape.n_dim != dims.size()) return false;
    size_t i = 0;
    for (const int dim : dims) if (shape.dim_size[i++] != static_cast<uint32_t>(dim)) return false;
    return true;
  }

  cv::Mat infer(const cv::Mat &bgr) {
    const int width = bgr.cols, height = bgr.rows;
    const float scale = std::min(static_cast<float>(kSide) / width, static_cast<float>(kSide) / height);
    const int resized_width = static_cast<int>(std::nearbyint(width * scale));
    const int resized_height = static_cast<int>(std::nearbyint(height * scale));
    const int left = (kSide - resized_width) / 2, top = (kSide - resized_height) / 2;
    cv::Mat resized, padded;
    cv::resize(bgr, resized, {resized_width, resized_height});
    cv::copyMakeBorder(resized, padded, top, kSide - resized_height - top, left,
                       kSide - resized_width - left, cv::BORDER_CONSTANT, {114, 114, 114});

    vknn_input input{};
    input.buffer = padded.data;
    input.size = kSide * kSide * 3;
    input.type = VKNN_TENSOR_UINT8;
    if (vknn_process_input(context_, &input, 1) != VKNN_SUCC ||
        vknn_inference(context_, nullptr) != VKNN_SUCC) {
      throw std::runtime_error("VKNN input or inference failed");
    }
    vknn_output output[2]{};
    output[0].buffer = prediction_.data();
    output[0].size = prediction_.size();
    output[0].user_allocated = true;
    output[0].original_tensor = true;
    output[1].buffer = prototype_.data();
    output[1].size = prototype_.size();
    output[1].user_allocated = true;
    output[1].original_tensor = true;
    if (vknn_process_output(context_, output, 2) != VKNN_SUCC) {
      throw std::runtime_error("VKNN output copy failed");
    }

    std::vector<Detection> candidates;
    for (int i = 0; i < kAnchors; ++i) {
      const float score = prediction_[4 * kAnchors + i];
      if (score < confidence_) continue;
      const float cx = prediction_[i], cy = prediction_[kAnchors + i];
      const float bw = prediction_[2 * kAnchors + i], bh = prediction_[3 * kAnchors + i];
      const float x1 = std::clamp((cx - bw / 2 - left) / scale, 0.f, static_cast<float>(width));
      const float y1 = std::clamp((cy - bh / 2 - top) / scale, 0.f, static_cast<float>(height));
      const float x2 = std::clamp((cx + bw / 2 - left) / scale, 0.f, static_cast<float>(width));
      const float y2 = std::clamp((cy + bh / 2 - top) / scale, 0.f, static_cast<float>(height));
      if (x2 - x1 > 1 && y2 - y1 > 1) candidates.push_back({i, score, {x1, y1, x2 - x1, y2 - y1}});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection &a, const Detection &b) { return a.score > b.score; });
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8U);
    std::vector<Detection> kept;
    for (const auto &candidate : candidates) {
      if (std::any_of(kept.begin(), kept.end(), [&](const Detection &previous) {
            return iou(candidate.box, previous.box) > nms_iou_;
          })) continue;
      kept.push_back(candidate);
      cv::Mat logits = cv::Mat::zeros(kProtoSide, kProtoSide, CV_32F);
      for (int channel = 0; channel < kProtos; ++channel) {
        const float coeff = prediction_[(5 + channel) * kAnchors + candidate.anchor];
        const cv::Mat plane(kProtoSide, kProtoSide, CV_32F,
                            prototype_.data() + channel * kProtoSide * kProtoSide);
        logits += coeff * plane;
      }
      cv::max(logits, -80, logits);
      cv::min(logits, 80, logits);
      cv::exp(-logits, logits);
      cv::divide(1.0, 1.0 + logits, logits);
      cv::Mat enlarged, restored;
      cv::resize(logits, enlarged, {kSide, kSide});
      cv::resize(enlarged(cv::Rect(left, top, resized_width, resized_height)), restored, {width, height});
      const int x = std::max(0, static_cast<int>(std::floor(candidate.box.x)));
      const int y = std::max(0, static_cast<int>(std::floor(candidate.box.y)));
      const int right = std::min(width, static_cast<int>(std::ceil(candidate.box.x + candidate.box.width)));
      const int bottom = std::min(height, static_cast<int>(std::ceil(candidate.box.y + candidate.box.height)));
      if (right > x && bottom > y) {
        const cv::Rect roi(x, y, right - x, bottom - y);
        cv::Mat binary = restored(roi) > mask_threshold_;
        cv::bitwise_or(mask(roi), binary, mask(roi));
      }
    }
    return mask;
  }

  void on_image(const sensor_msgs::msg::Image &image) {
    if (image.width == 0 || image.height == 0 || image.step < image.width * 3 ||
        image.data.size() < static_cast<size_t>(image.step) * image.height ||
        (image.encoding != "bgr8" && image.encoding != "rgb8")) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Expected nonempty bgr8/rgb8 image");
      return;
    }
    if (camera_width_ && (camera_width_ != image.width || camera_height_ != image.height)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "CameraInfo and image dimensions differ");
    }
    const auto begin = std::chrono::steady_clock::now();
    try {
      cv::Mat input(image.height, image.width, CV_8UC3, const_cast<uint8_t *>(image.data.data()), image.step);
      cv::Mat bgr;
      if (image.encoding == "rgb8") cv::cvtColor(input, bgr, cv::COLOR_RGB2BGR);
      else bgr = input;
      const cv::Mat mask = infer(bgr);
      sensor_msgs::msg::Image result;
      result.header = image.header;
      result.height = image.height;
      result.width = image.width;
      result.encoding = "mono8";
      result.is_bigendian = false;
      result.step = image.width;
      result.data.assign(mask.datastart, mask.dataend);
      mask_pub_->publish(std::move(result));
    } catch (const std::exception &error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Grass inference failed: %s", error.what());
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    latencies_.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    if (latencies_.size() < static_cast<size_t>(stats_interval_)) return;
    std::sort(latencies_.begin(), latencies_.end());
    const auto percentile = [&](double p) {
      return latencies_[static_cast<size_t>(std::ceil(p * latencies_.size())) - 1];
    };
    const double seconds = std::chrono::duration<double>(end - window_start_).count();
    RCLCPP_INFO(get_logger(), "frames=%zu fps=%.2f p50_ms=%.2f p95_ms=%.2f rss_kib=%zu",
                latencies_.size(), latencies_.size() / seconds, percentile(0.5), percentile(0.95), rss_kib());
    latencies_.clear();
    window_start_ = end;
  }

  vknn_context context_{};
  std::vector<float> prediction_, prototype_, latencies_;
  double confidence_{}, nms_iou_{}, mask_threshold_{};
  int stats_interval_{};
  uint32_t camera_width_{}, camera_height_{};
  std::chrono::steady_clock::time_point window_start_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<GrassSegNode>());
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("grass_seg_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
