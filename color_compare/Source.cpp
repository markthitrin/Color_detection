#include "opencv2/opencv.hpp"
#include "iostream"
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#define PI 3.14159265358979323846

cv::VideoCapture camera(0);
cv::Mat frame;

class indicator {
public:
    std::string name;
    cv::Vec3b color;
};

std::vector<indicator> Ind;

void load_indicator(const std::string& input_file_name,std::vector<indicator>& Ind) {
    std::ifstream input_file(input_file_name);
    while (!input_file.eof()) {
        indicator _Ind;
        std::getline(input_file, _Ind.name);
        input_file >> _Ind.color[0] >> _Ind.color[1] >> _Ind.color[2];
        Ind.push_back(_Ind);
    }
}

void put_indicator(const std::string& output_file_name, std::vector<indicator>& Ind) {
    std::ofstream output_file(output_file_name);
    for (int i = 0; i < Ind.size(); i++) {
        output_file << Ind[i].name << "\n";
        output_file << Ind[i].color[0] << " " << Ind[i].color[1] << " " << Ind[i].color[2] << "\n";
    }
}



double get_distance(const cv::Vec3b& p1,const cv::Vec3b& p2) { // LAB color space

    const double KL = 1;
    const double KC = 1;
    const double KH = 1;

    auto get_LCh = [](cv::Vec3b p) {
        cv::Vec3b result;
        result[0] = p[0]; // L
        result[1] = std::sqrt(std::pow(p[1], 2) + std::pow(p[2], 2));// C
        result[2] = std::atan2(p[2], p[1]) * double(180) / PI;
        return result;
    };

    cv::Vec3b p1LCh = get_LCh(p1);
    cv::Vec3b p2LCh = get_LCh(p2);

    double avrC = (p1LCh[1] + p2LCh[2]) / double(2);
    double G = 0.5 * (double(1) - std::sqrt(std::pow(avrC, 7) / (std::pow(avrC, 7) + std::pow(25, 7))));

    double a1_ = (1 + G) * p1[1];
    double a2_ = (1 + G) * p2[1];

    double C1_ = std::sqrt(std::pow(a1_, 2) + std::pow(p1[2], 2));
    double C2_ = std::sqrt(std::pow(a2_, 2) + std::pow(p2[2], 2));

    double h1_ = (p1[2] == a1_ && a1_ == 0) ? 0 : std::atan2(p1[2], a1_);
    double h2_ = (p2[2] == a2_ && a2_ == 0) ? 0 : std::atan2(p2[2], a2_);

    double dL = p2[0] - p1[0];
    double dC = C2_ - C1_;
    double dh = C1_ * C2_ == 0 ? 0 :
        std::abs(h2_ - h1_) <= 180 ? h2_ - h1_ :
        h2_ - h1_ > 180 ? h2_ - h1_ - 360 :
        h2_ - h1_ + 360;

    double dH = 2 * std::sqrt(C1_ * C2_) * std::sin(dh / 2);

    double avrL = (p1LCh[0] + p2LCh[1]) / 2;
    double avrC_ = (C1_ + C2_) / 2;
    double avrh = C1_ * C2_ == 0 ? h1_ + h2_ :
        std::abs(h1_ - h2_) <= 180 ? (h1_ + h2_) / 2 :
        h1_ + h2_ < 360 ? (h1_ + h2_ + 360) / 2 :
        (h1_ + h2_ - 360) / 2;

    double T = double(1)
        - 0.17 * std::cos(avrh - 30)
        + 0.24 * std::cos(2 * avrh)
        + 0.32 * std::cos(3 * avrh + 6)
        - 0.20 * std::cos(4 * avrh - 63);

    double dangle = 30 * std::exp(-(avrh - 275) / 25);
    double Rc = 2 * std::sqrt(std::pow(avrC_, 7) / (std::pow(avrC_, 7) + std::pow(25, 7)));

    double SL = double(1) + (0.015) * std::pow(avrL - 50, 2) / std::sqrt(20 + std::pow(avrL - 50, 2));
    double SC = double(1) + 0.045 * avrC_;
    double SH = double(1) + 0.015 * avrC_ * T;
    double RT = -std::sin(2 * dangle * PI / 180) * Rc;

    double result = std::sqrt(
        std::pow(dL / (KL * SL), 2)
        + std::pow(dC / (KC * SC), 2)
        + std::pow(dH / (KH * SH), 2)
        + RT * (dC / (KC * SC)) * (dH / (KH * SH)));
    return result;
}

inline void invert_color(cv::Vec3b& point) {
    point[0] = 255 - point[0];
    point[1] = 255 - point[1];
    point[2] = 255 - point[2];
}

void draw_capture_box(cv::Mat& image) {
    const int row = image.rows;
    const int col = image.cols;

    int midx = col / 2;
    int midy = row / 2;
    int edge_size = ((row + col) / 2) / 4;

    int left = midx - edge_size / 2;
    int right = midx + edge_size / 2;
    int top = midy + edge_size / 2;
    int bottom = midy - edge_size / 2;
    for (int q = left; q <= right; q++) {
        invert_color(image.at<cv::Vec3b>(top, q));
        invert_color(image.at<cv::Vec3b>(bottom, q));
    }
    for (int q = bottom; q <= top; q++) {
        invert_color(image.at<cv::Vec3b>(q, left));
        invert_color(image.at<cv::Vec3b>(q, right));
    }
}

cv::Vec3b get_color_capture_box(cv::Mat& image) {
    const int row = image.rows;
    const int col = image.cols;

    int midx = col / 2;
    int midy = row / 2;
    int edge_size = ((row + col) / 2) / 4;

    int left = midx - edge_size / 2;
    int right = midx + edge_size / 2;
    int top = midy + edge_size / 2;
    int bottom = midy - edge_size / 2;

    double get_color[3] = {128,0,0};
    int pixel_count = 0;
    for (int q = left; q <= right; q++) {
        for (int w = bottom; w <= top; w++) {
            get_color[0] += image.at<cv::Vec3b>(q, w)[0];
            get_color[1] += image.at<cv::Vec3b>(q, w)[1];
            get_color[2] += image.at<cv::Vec3b>(q, w)[2];
            ++pixel_count;
        }
    }
    cv::Vec3b return_color;
    return_color[0] = get_color[0] / pixel_count;
    return_color[1] = get_color[1] / pixel_count;
    return_color[2] = get_color[2] / pixel_count;
    return return_color;
}

bool is_prefix(const std::string& comparator,const std::string& str,int& pointer) {
    int _pointer = pointer;
    for (int i = 0; i < comparator.size(); i++) {
        if (str[_pointer] != comparator[i])
            return false;
        ++_pointer;
    }
    pointer = _pointer;
    return true;
}

void execute_command(const std::string& str) {
    int pointer = 0;
    if (is_prefix("capture ", str, pointer)) {
        Ind.push_back({ "color123",get_color_capture_box(frame) });
    }
}

void show() {
    if (!camera.isOpened()) {
        std::cout << "asdfasdfasdfasdf\n\n\n\n\n\n\n\n\n";
    }
    camera >> frame;
    draw_capture_box(frame);
    cv::imshow("Webcam", frame);

    cv::Mat output_color = frame;
    cv::Vec3b get_color = get_color_capture_box(frame);
    for (int i = 0; i < output_color.rows; i++) {
        for (int j = 0; j < output_color.cols; j++) {
            output_color.at<cv::Vec3b>(i, j) = get_color;
        }
    }
    cv::imshow("Color_captured", output_color);

}

int main(int, char**) {
    cv::VideoCapture camera(0);
    cv::namedWindow("Webcam", 1080);
    cv::namedWindow("Color_captured", 1080);

    camera >> frame;
    cv::imshow("Wdsfadsf", frame);
    show();
    
    while (true) {
        std::string command;
        std::getline(std::cin, command);
        execute_command(command);
    }

    return 0;
}