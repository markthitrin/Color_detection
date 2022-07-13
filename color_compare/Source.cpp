#include "opencv2/opencv.hpp"
#include "iostream"
#include <chrono>
#define PI 3.14159265358979323846

double get_distance(const cv::Vec3b& p1,const cv::Vec3b& p2) { // LAB color space
    const double KL = 1;
    const double KC = 1;
    const double KH = 1;

    auto get_LCh = [](const cv::Vec3b& p) {
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

int main(int, char**) {
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    cv::namedWindow("Webcam", 1080);
    cv::Mat frame;

    

    while (1) {
        camera >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2);
        for (int q = 0; q < frame.rows; q++) {
            for (int w = 0; w < frame.cols; w++) {
                cv::Vec3b p = frame.at<cv::Vec3b>(q, w);
                p[0] -= p[0]%40;
                frame.at<cv::Vec3b>(q, w) = p;
            }
        }
        cv::cvtColor(frame, frame, cv::COLOR_HSV2BGR);
        cv::imshow("Webcam", frame);
        if (cv::waitKey(10) >= 0)
            break;
    }
    return 0;
}