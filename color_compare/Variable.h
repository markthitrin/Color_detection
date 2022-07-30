#pragma once
#include "Header.h"
#include "indicator.h"

extern bool get_command_terminate;
extern cv::VideoCapture camera;
extern cv::Mat frame;
extern cv::Mat pure_frame;
extern cv::Mat get_camera;
extern cv::Mat get_color_capture;
extern cv::Mat get_color_detected;

extern std::atomic<int> waiting_count;
extern bool show_camera_wait;
extern bool show_color_capture_wait;
extern bool show_color_detected_wait;

extern std::vector<indicator> Ind;
extern std::vector<indicator> color_base;

extern std::function<double(const Matrix<double>&, const Matrix<double>&)> mean_squre_loss_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dmean_squre_loss_func;