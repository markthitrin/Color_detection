#pragma once
#include "Header.h"
#include "indicator.h"

extern double red_ratio;
extern double green_ratio;
extern double blue_ratio;

extern double gamma_red;
extern double gamma_green;
extern double gamma_blue;

extern double gamma_mul_red;
extern double gamma_mul_green;
extern double gamma_mul_blue;

extern bool havetuned;

extern Neural_Network CCM;
extern Neural_Network red_poly_trans;
extern Neural_Network green_poly_trans;
extern Neural_Network blue_poly_trans;

void init(std::vector<indicator> color_base, std::vector<cv::Vec3b> src);

cv::Vec3b tune(cv::Vec3b color_base);

void tune(cv::Mat& frame);