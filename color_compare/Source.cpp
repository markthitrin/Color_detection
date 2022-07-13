#include "opencv2/opencv.hpp"
#include "iostream"
#include <chrono>


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
        cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
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