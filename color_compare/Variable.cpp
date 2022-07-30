#pragma once
#include "Header.h"
#include "indicator.h"

bool get_command_terminate = true;
cv::VideoCapture camera(0);
cv::Mat frame;
cv::Mat pure_frame;
cv::Mat get_camera;
cv::Mat get_color_capture;
cv::Mat get_color_detected;

std::atomic<int> waiting_count = 0;
bool show_camera_wait = true;
bool show_color_capture_wait = true;
bool show_color_detected_wait = true;

std::vector<indicator> Ind;
std::vector<indicator> color_base;


std::function<double(const Matrix<double>&,const Matrix<double>&)> mean_squre_loss_func = [](const Matrix<double>& input, const Matrix<double>& target) {
    double result = 0;
    for (int i = 0; i < target.get_row(); i++) {
        for (int j = 0; j < target.get_column(); j++) {
            result += std::pow(target[i][j] - input[i][j], 2);
        }
    }
    return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dmean_squre_loss_func = [](const Matrix<double>& input, const Matrix<double>& target) {
    Matrix<double> result(target);
    for (int i = 0; i < target.get_row(); i++) {
        for (int j = 0; j < target.get_column(); j++) {
            result[i][j] = 5 * (target[i][j] - input[i][j]);
        }
    }
    return result;
};