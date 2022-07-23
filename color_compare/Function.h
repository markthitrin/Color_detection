#pragma once
#include "Header.h"
#include "indicator.h"

void init(std::vector<indicator> color_base, std::vector<cv::Vec3b> src);

cv::Vec3b tune(cv::Vec3b color_base);

void tune(cv::Mat& frame);