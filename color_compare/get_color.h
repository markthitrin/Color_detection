#pragma once
#include "Header.h"

cv::Vec3b get_mean_color(cv::Mat image, int left, int right, int bottom, int top);

cv::Vec3b get_squremean_color(cv::Mat image, int left, int right, int bottom, int top);