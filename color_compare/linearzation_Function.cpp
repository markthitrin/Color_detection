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
Neural_Network red_poly_trans;
Neural_Network green_poly_trans;
Neural_Network blue_poly_trans;

void set_ratio(double(&_color_base)[24][3], double(&_src)[24][3]) {
	blue_ratio = _color_base[18][0] / _src[18][0];
	green_ratio = _color_base[18][1] / _src[18][1];
	red_ratio = _color_base[18][2] / _src[18][2];

	for (int i = 0; i < 24; i++) {
		_src[i][0] *= blue_ratio;
		_src[i][1] *= green_ratio;
		_src[i][2] *= red_ratio;
	}
}

void set_gamma(double(&_color_base)[24][3], double(&_src)[24][3]) {
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
		_src[i][1] = std::pow(_src[i][1] / gamma_mul_green, gamma_green) * gamma_mul_green;
		_src[i][0] = std::pow(_src[i][0] / gamma_mul_blue, gamma_blue) * gamma_mul_blue;
	}
}

void set_polynomial(double(&_color_base)[24][3], double(&_src)[24][3], const int learn_time){
	std::ofstream output_red("outputred.txt");
	auto to_Matrix = [](const double v,const int size) {
		Matrix<double> result(size, 1);
		for (int i = 0; i < size; i++) {
			result[i][0] = std::pow(v, i);
		}
		return result;
	};
	for (int learning_time = 0; learning_time < learn_time; learning_time++) {
		
		auto to_Matrix1 = [](const double v) {
			Matrix<double> result(1, 1);
			result[0][0] = v;
			return result;
		};

		double red_loss = 0;
		double green_loss = 0;
		double blue_loss = 0;

		red_poly_trans.set_change_dependencies(0);
		for (int i = 18; i < 24; i++) {
			red_poly_trans.feedforward(to_Matrix(_src[i][2], 3));
			red_poly_trans.backpropagation(to_Matrix1(_color_base[i][2]));
			red_loss += red_poly_trans.get_loss(to_Matrix1(_color_base[i][2]));
		}
		red_loss /= 6;
		//red_poly_trans.set_all_drop_out_rate(3e-14 * red_loss / 20000);
		red_poly_trans.change_dependencies();
		red_poly_trans.forgot_all();
		output_red << red_loss << "\n";

		green_poly_trans.set_change_dependencies(0);
		for (int i = 18; i < 24; i++) {
			green_poly_trans.feedforward(to_Matrix(_src[i][1], 3));
			green_poly_trans.backpropagation(to_Matrix1(_color_base[i][1]));
			green_loss += green_poly_trans.get_loss(to_Matrix1(_color_base[i][1]));
		}
		green_loss /=6;
		//green_poly_trans.set_all_drop_out_rate(3e-14 * green_loss / 20000);
		green_poly_trans.change_dependencies();
		green_poly_trans.forgot_all();

		blue_poly_trans.set_change_dependencies(0);
		for (int i = 18; i < 24; i++) {
			blue_poly_trans.feedforward(to_Matrix(_src[i][0], 3));
			blue_poly_trans.backpropagation(to_Matrix1(_color_base[i][0]));
			blue_loss += blue_poly_trans.get_loss(to_Matrix1(_color_base[i][0]));
		}
		blue_loss /= 6;
		//blue_poly_trans.set_all_drop_out_rate(3e-14 * blue_loss / 20000);
		blue_poly_trans.change_dependencies();
		blue_poly_trans.forgot_all();
	}

	for (int i = 0; i < 24; i++) {
		red_poly_trans.feedforward(to_Matrix( _src[i][2] ,3));
		red_poly_trans.forgot_all();
		_src[i][2] = red_poly_trans.get_output()[0][0];

		green_poly_trans.feedforward(to_Matrix( _src[i][1] ,3));
		green_poly_trans.forgot_all();
		_src[i][1] = green_poly_trans.get_output()[0][0];

		blue_poly_trans.feedforward(to_Matrix( _src[i][0] ,3));
		blue_poly_trans.forgot_all();
		_src[i][0] = blue_poly_trans.get_output()[0][0];
	}
}

void set_CCM(double(&_color_base)[24][3], double(&_src)[24][3], const int learn_time) {
	auto to_Matrix = [](const double v[], int size) {
		Matrix<double> result(size, 1);
		for (int i = 0; i < result.get_row(); i++) {
			result[i][0] = v[i];
		}
		return result;
	};
	for (int learning_time = 0; learning_time < learn_time; learning_time++) {
		
		double loss = 0;

		CCM.set_change_dependencies(0);
		for (int i = 0; i < 24; i++) {
			CCM.feedforward(to_Matrix(_src[i], 3));
			CCM.backpropagation(to_Matrix(_color_base[i], 3));
			loss += CCM.get_loss(to_Matrix(_color_base[i], 3));
		}
		loss /= 24;
		//CCM.set_all_drop_out_rate(3e-12 * loss / 20000);
		CCM.change_dependencies();
		CCM.forgot_all();
	}

	for (int i = 0; i < 24; i++) {
		CCM.feedforward(to_Matrix(_src[i], 3));
		CCM.forgot_all();
		_src[i][0] = CCM.get_output()[0][0];
		_src[i][1] = CCM.get_output()[1][0];
		_src[i][2] = CCM.get_output()[2][0];
	}


}

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
	std::cout << "not tune\n";
	for (int i = 0; i < 24; i++) {
		for (int w = 0; w < 3; w++) {
			std::cout << (int)_src[i][w] << "\n";
		}
	}

	// set white
	set_ratio(_color_base, _src);
	std::cout << "ratio tune\n";
	for (int i = 0; i < 24; i++) {
		for (int w = 0; w < 3; w++) {
			std::cout << (int)_src[i][w] << "\n";
		}
	}

	// gamma tune, using ML
	set_gamma(_color_base, _src);
	std::cout << "gamma tune\n";
	for (int i = 0; i < 24; i++) {
		for (int w = 0; w < 3; w++) {
			std::cout << (int)_src[i][w] << "\n";
		}
	}

	// set polynomail
	set_polynomial(_color_base, _src, 10000);
	std::cout << "poly tune\n";
	for (int i = 0; i < 24; i++) {
		for (int w = 0; w < 3; w++) {
			std::cout << (int)_src[i][w] << "\n";
		}
	}
	
	// set CCM
	set_CCM(_color_base, _src, 10000);
	std::cout << "CCM tune";
	for (int i = 0; i < 24; i++) {
		for (int w = 0; w < 3; w++) {
			std::cout << (int)_src[i][w] << "\n";
		}
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

	auto to_Matrix_poly = [](const double v, const int size) {
		Matrix<double> result(size, 1);
		for (int i = 0; i < size; i++) {
			result[i][0] = std::pow(v, i);
		}
		return result;
	};

	if (havetuned) {
		red_poly_trans.feedforward(to_Matrix_poly(red,3));
		red_poly_trans.forgot_all();
		red = red_poly_trans.get_output()[0][0];
		
		green_poly_trans.feedforward(to_Matrix_poly(green,3));
		green_poly_trans.forgot_all();
		green = green_poly_trans.get_output()[0][0];

		blue_poly_trans.feedforward(to_Matrix_poly(blue,3));
		blue_poly_trans.forgot_all();
		blue = blue_poly_trans.get_output()[0][0];

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
	cv::cvtColor(frame, frame, cv::COLOR_BGR2Lab);
	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			frame.at<cv::Vec3b>(i,j) = tune(frame.at<cv::Vec3b>(i, j));
		}
	}
	cv::cvtColor(frame, frame, cv::COLOR_Lab2BGR);
}