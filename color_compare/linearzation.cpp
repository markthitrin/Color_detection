#pragma once
#include "Header.h"
#include "indicator.h"

double red_ratio = 1;
double green_ratio = 1;
double blue_ratio = 1;

double gamma_red = 1;
double gamma_green = 1;
double gamma_blue = 1;

double gamma_mul_red = 255;
double gamma_mul_green = 255;
double gamma_mul_blue = 255;

bool havetuned = false;

Neural_Network CCM;

void init(std::vector<indicator> color_base, std::vector<cv::Vec3b> src) {
	double _src[24][3];
	for (int i = 0; i < src.size(); i++) {
		_src[i][0] = (double)src[i][0];
		_src[i][1] = (double)src[i][1];
		_src[i][2] = (double)src[i][2];
	}

	double _color_base[24][3];
	for (int i = 0; i < color_base.size(); i++) {
		_color_base[i][0] = (double)color_base[i].color[0];
		_color_base[i][1] = (double)color_base[i].color[1];
		_color_base[i][2] = (double)color_base[i].color[2];
	}

	// set white
	blue_ratio = _color_base[18][0] / _src[18][0];
	green_ratio = _color_base[18][1] / _src[18][1];
	red_ratio = _color_base[18][2] / _src[18][2];

	for (int i = 0; i < src.size(); i++) {
		_src[i][0] *= blue_ratio;
		_src[i][1] *= green_ratio;
		_src[i][2] *= red_ratio;
	}

	// gamma tune, using ML
	for (int learning_time = 0; learning_time < 100000; learning_time++) {
		// error -> mean squre;
		const double learning_rate_red = 0.0001;
		double error_red = 0;
		double error_mul_red = 0;
		for (int i = 19; i < 24; i++) {
			error_red += 2 * (_color_base[i][2] / gamma_mul_red - std::pow(_src[i][2] / gamma_mul_red, gamma_red)) * log(_src[i][2] / gamma_mul_red) * learning_rate_red;
		}
		for (int i = 19; i < 24; i++) {
			error_mul_red += 2 * (_color_base[i][2] / gamma_mul_red - std::pow(_src[i][2] / gamma_mul_red, gamma_red)) * std::pow(_src[i][2], gamma_red - 1) * (gamma_red / gamma_mul_red) * learning_rate_red;
		}

		const double learning_rate_green = 0.0001;
		double error_green = 0;
		double error_mul_green = 0;
		for (int i = 19; i < 24; i++) {
			error_green += 2 * (_color_base[i][1] / gamma_mul_green - std::pow(_src[i][1] / gamma_mul_green, gamma_green)) * log(_src[i][1] / gamma_mul_green) * learning_rate_green;
		}
		for (int i = 19; i < 24; i++) {
			error_mul_green += 2 * (_color_base[i][1] / gamma_mul_green - std::pow(_src[i][1] / gamma_mul_green, gamma_green)) * std::pow(_src[i][1], gamma_green - 1) * (gamma_green / gamma_mul_green) * learning_rate_green;
		}

		const double learning_rate_blue = 0.0001;
		double error_blue = 0;
		double error_mul_blue = 0;
		for (int i = 19; i < 24; i++) {
			error_blue += 2 * (_color_base[i][0] / gamma_mul_blue - std::pow(_src[i][0] / gamma_mul_blue, gamma_blue)) * log(_src[i][0] / gamma_mul_blue) * learning_rate_blue;
		}
		for (int i = 19; i < 24; i++) {
			error_mul_blue += 2 * (_color_base[i][0] / gamma_mul_blue - std::pow(_src[i][0] / gamma_mul_blue, gamma_blue)) * std::pow(_src[i][0], gamma_blue - 1) * (gamma_blue / gamma_mul_blue) * learning_rate_blue;
		}

		gamma_red += error_red;
		gamma_green += error_green;
		gamma_blue += error_blue;
	}

	for (int i = 0; i < 24; i++) {
		_src[i][2] = std::pow(_src[i][2] / gamma_mul_red, gamma_red) * gamma_mul_red;
		_src[i][1] = std::pow(_src[i][1] / gamma_mul_green, gamma_green)* gamma_mul_green;
		_src[i][0] = std::pow(_src[i][0] / gamma_mul_blue, gamma_blue) * gamma_mul_blue;
	}

	std::ofstream output_ai("output_ai.txt");
	for (int learning_time = 0; learning_time < 2000; learning_time++) {
		auto to_Matrix = [](const double v[],int size) {
			Matrix<double> result(size, 1);
			for (int i = 0; i < result.get_row(); i++) {
				result[i][0] = v[i];
			}
			return result;
		};
		double loss = 0;
		
		CCM.set_change_dependencies(0);
		for (int i = 0; i < 24; i++) {
			CCM.feedforward(to_Matrix(_src[i], 3));
			CCM.backpropagation(to_Matrix(_color_base[i], 3));
			loss += CCM.get_loss(to_Matrix(_color_base[i], 3));
		}
		loss /= 24;
		output_ai << loss << "\n";
		CCM.set_all_drop_out_rate(3e-7 * loss / 20000);
		CCM.change_dependencies();
		CCM.forgot_all();
	}


	havetuned = true;
}

cv::Vec3b tune(cv::Vec3b color_base) {
	cv::Vec3b result;

	double blue = color_base[0];
	double green = color_base[1];
	double red = color_base[2];

	red *= red_ratio;
	green *= green_ratio;
	blue *= blue_ratio;

	red = std::pow(red / gamma_mul_red, gamma_red) * gamma_mul_red;
	green = std::pow(green / gamma_mul_green, gamma_green) * gamma_mul_green;
	blue = std::pow(blue / gamma_mul_blue, gamma_blue) * gamma_mul_blue;

	auto to_Matrix = [](std::vector<double> v) {
		Matrix<double> result(v.size(), 1);
		for (int i = 0; i < result.get_row(); i++) {
			result[i][0] = v[i];
		}
		return result;
	};

	if (havetuned) {
		CCM.feedforward(to_Matrix({ blue,green,red }));
		CCM.forgot_all();
		blue = CCM.get_output()[0][0];
		green = CCM.get_output()[1][0];
		red = CCM.get_output()[2][0];
	}

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