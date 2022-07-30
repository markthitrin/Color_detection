#pragma once
#include "Header.h"

cv::Vec3b get_mean_color(cv::Mat image, int left, int right, int bottom, int top) {
	if (bottom > top) std::swap(bottom, top);
	if (left > right) std::swap(left, top);
	double c1 = 0, c2 = 0, c3 = 0;
	double count = 0;
	cv::Vec3b result;
	for (int i = bottom; i <= top; i++) {
		for (int j = left; j <= right; j++) {
			c1 += image.at<cv::Vec3b>(i, j)[0];
			c2 += image.at<cv::Vec3b>(i, j)[1];
			c3 += image.at<cv::Vec3b>(i, j)[2];
			++count;
		}
	}
	c1 /= count; c2 /= count; c3 /= count;
	result[0] = int(c1);
	result[1] = int(c2);
	result[2] = int(c3);
	return result;
}

cv::Vec3b get_squremean_color(cv::Mat image, int left, int right, int bottom, int top) {
	if (bottom > top) std::swap(bottom, top);
	if (left > right) std::swap(left, top);
	double c1 = 0, c2 = 0, c3 = 0;
	double count = 0;
	cv::Vec3b result;
	for (int i = bottom; i <= top; i++) {
		for (int j = left; j <= right; j++) {
			c1 += std::pow((double)image.at<cv::Vec3b>(i, j)[0],2);
			c2 += std::pow((double)image.at<cv::Vec3b>(i, j)[1],2);
			c3 += std::pow((double)image.at<cv::Vec3b>(i, j)[2],2);
			++count;
		}
	}
	c1 /= count; c2 /= count; c3 /= count;
	c1 = std::sqrt(c1); c2 = std::sqrt(c2); c3 = std::sqrt(c3);
	result[0] = int(c1);
	result[1] = int(c2);
	result[2] = int(c3);
	return result;
}