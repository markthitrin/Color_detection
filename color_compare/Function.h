#pragma once
#include "Header.h"
#include "indicator.h"

void load_indicator(const std::string& input_file_name, std::vector<indicator>& Ind);

void put_indicator(const std::string& output_file_name, std::vector<indicator>& Ind);

cv::Vec3b BRG2Lab(cv::Vec3b p);

double to_rad(double degree);

double get_distance(cv::Vec3d lab1, cv::Vec3d lab2, double kL = 1.0, double kC = 1.0, double kH = 1.0);

inline void invert_color(cv::Vec3b& point);

void flip_image(cv::Mat& image);

void draw_capture_box0(cv::Mat& image);

void draw_capture_box(cv::Mat& image);

std::vector<cv::Vec3b> get_color_calibrator(cv::Mat& image);

cv::Vec3b get_color_capture_box(cv::Mat& image);

bool is_command(const std::string& comparator, const std::string& str, int& pointer);

void get_frame();

void execute_command(const std::string& str);

void get_command();

void show_camera();

void show_capture_color();

void show_color_detected();