#pragma once
#include "Header.h"
#include "indicator.h"

double red_ratio = 1;
double green_ratio = 1;
double blue_ratio = 1;

double gamma_red = 1;
double gamma_green = 1;
double gamma_blue = 1;

void init(std::vector<indicator> color_base,std::vector<cv::Vec3b> src) {
	// set white
	blue_ratio = double(color_base[18].color[0]) / double(src[18][0]);
	green_ratio = double(color_base[18].color[1]) / double(src[18][1]);
	red_ratio = double(color_base[18].color[2]) / double(src[18][2]);

	for (int i = 0; i < src.size(); i++) {
		src[i][0] *= blue_ratio;
		src[i][1] *= green_ratio;
		src[i][2] *= red_ratio;
	}

	// gamma tune, using ML
	for (int learning_time = 0; learning_time < 100000; learning_time++) {
		// error -> mean squre;
		const double learning_rate_red = 0.001;
		double error_red = 0;
		for (int i = 19; i < 24; i++) {
			error_red += 2 * ((double)color_base[i].color[2] / 255 - std::pow((double)src[i][2] / 255, gamma_red)) * log((double)src[i][2] / 255) * learning_rate_red;
		}

		const double learning_rate_green = 0.001;
		double error_green = 0;
		for (int i = 19; i < 24; i++) {
			error_green += 2 * ((double)color_base[i].color[1] / 255 - std::pow((double)src[i][1] / 255, gamma_green)) * log((double)src[i][1] / 255) * learning_rate_green;
		}

		const double learning_rate_blue = 0.001;
		double error_blue = 0;
		for (int i = 19; i < 24; i++) {
			error_blue += 2 * ((double)color_base[i].color[0] / 255 - std::pow((double)src[i][0] / 255, gamma_blue)) * log((double)src[i][0] / 255) * learning_rate_blue;
		}

		gamma_red += error_red;
		gamma_green += error_green;
		gamma_blue += error_blue;
	}
}

cv::Vec3b tune(cv::Vec3b color_base) {
	cv::Vec3b result;

	double blue = color_base[0];
	double green = color_base[1];
	double red = color_base[2];

	red *= red_ratio;
	green *= green_ratio;
	blue *= blue_ratio;

	red = std::pow(red / 255, gamma_red) * 255;
	green = std::pow(green / 255, gamma_green) * 255;
	blue = std::pow(blue / 255, gamma_blue) * 255;

	if (blue > 255)
		blue = 255;
	else if (blue < 0)
		blue = 0;

	if (green > 255)
		green = 255;
	else if (green < 0)
		green = 0;

	if (red > 255)
		red = 255;
	else if (red < 0)
		red = 0;

	result[0] = blue;
	result[1] = green;
	result[2] = red;
	return result;
}

void tune(cv::Mat& frame) {
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			frame.at<cv::Vec3b>(i,j) = tune(frame.at<cv::Vec3b>(i, j));
		}
	}
}