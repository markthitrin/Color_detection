#pragma once
#include "Variable.h"
#include "indicator.h"
#include "linearzation_Function.h"

void load_indicator(const std::string& input_file_name, std::vector<indicator>& Ind) {
    std::ifstream input_file(input_file_name);
    while (!input_file.eof()) {
        indicator _Ind;
        std::getline(input_file, _Ind.name);
        double B, G, R;
        input_file >> B >> G >> R;
        _Ind.color[0] = B;
        _Ind.color[1] = G;
        _Ind.color[2] = R;
        std::string get_;
        std::getline(input_file, get_);
        Ind.push_back(_Ind);
    }
}

void put_indicator(const std::string& output_file_name, std::vector<indicator>& Ind) {
    std::ofstream output_file(output_file_name);
    for (int i = 0; i < Ind.size(); i++) {
        output_file << Ind[i].name << "\n";
        output_file << int(Ind[i].color[0]) << " " << int(Ind[i].color[1]) << " " << int(Ind[i].color[2]) << "\n";
    }
}

cv::Vec3b BRG2Lab(cv::Vec3b p) {
    auto f = [](double x) {
        return x > 0.008856 ? std::pow(x, 0.33) : 7.787 * x + double(16) / 166;
    };

    double b = p[0];
    double g = p[1];
    double r = p[2];
    b /= 255;
    g /= 255;
    r /= 255;
    double x = 0.412453 * r + 0.357580 * g + 0.180423 * b;
    double y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
    double z = 0.019334 * r + 0.119193 * g + 0.950227 * b;

    x = x / 0.950456;
    z = z / 1.088754;

    double l = y > 0.008856 ? double(116) * std::pow(y, 0.33) - 16 : 903.3 * y;
    double A = 500 * (f(x) - f(y));
    double B = 200 * (f(y) - f(z));

    cv::Vec3b result;
    result[0] = l * double(255) / 100;
    result[1] = A + 128;
    result[2] = B + 128;
    return result;
}

double to_rad(double degree) {
    return degree / 180 * CV_PI;
}

double get_distance(cv::Vec3d lab1, cv::Vec3d lab2, double kL = 1.0, double kC = 1.0, double kH = 1.0) {
    double delta_L_apo = lab2[0] - lab1[0];
    double l_bar_apo = (lab1[0] + lab2[0]) / 2.0;
    double C1 = sqrt(pow(lab1[1], 2) + pow(lab1[2], 2));
    double C2 = sqrt(pow(lab2[1], 2) + pow(lab2[2], 2));
    double C_bar = (C1 + C2) / 2.0;
    double G = sqrt(pow(C_bar, 7) / (pow(C_bar, 7) + pow(25, 7)));
    double a1_apo = lab1[1] + lab1[1] / 2.0 * (1.0 - G);
    double a2_apo = lab2[1] + lab2[1] / 2.0 * (1.0 - G);
    double C1_apo = sqrt(pow(a1_apo, 2) + pow(lab1[2], 2));
    double C2_apo = sqrt(pow(a2_apo, 2) + pow(lab2[2], 2));
    double C_bar_apo = (C1_apo + C2_apo) / 2.0;
    double delta_C_apo = C2_apo - C1_apo;
    //h1 and h2
    double h1_apo;
    if (C1_apo == 0) {
        h1_apo = 0.0;
    }
    else {
        h1_apo = atan2(lab1[2], a1_apo);
        if (h1_apo < 0.0) h1_apo += 2. * CV_PI;
    }
    double h2_apo;
    if (C2_apo == 0) {
        h2_apo = 0.0;
    }
    else {
        h2_apo = atan2(lab2[2], a2_apo);
        if (h2_apo < 0.0) h2_apo += 2. * CV_PI;
    }
    //delta_h_apo
    double delta_h_apo;
    if (abs(h2_apo - h1_apo) <= CV_PI)
    {
        delta_h_apo = h2_apo - h1_apo;
    }
    else if (h2_apo <= h1_apo)
    {
        delta_h_apo = h2_apo - h1_apo + 2. * CV_PI;
    }
    else
    {
        delta_h_apo = h2_apo - h1_apo - 2. * CV_PI;
    }
    //H_apo
    double H_bar_apo;
    if (C1_apo == 0 || C2_apo == 0) {
        H_bar_apo = h1_apo + h2_apo;
    }
    else if (abs(h1_apo - h2_apo) <= CV_PI) {
        H_bar_apo = (h1_apo + h2_apo) / 2.0;
    }
    else if (h1_apo + h2_apo < 2. * CV_PI) {
        H_bar_apo = (h1_apo + h2_apo + 2. * CV_PI) / 2.0;
    }
    else {
        H_bar_apo = (h1_apo + h2_apo - 2. * CV_PI) / 2.0;
    }
    //delta_H_apo
    double delta_H_apo = 2.0 * sqrt(C1_apo * C2_apo) * sin(delta_h_apo / 2.0);


    //double delta_H;

    //delta_H_apo = 2.0 * sqrt(C1 * C2) * sin(delta_h_apo / 2.0);
    double T = 1.0 - 0.17 * cos(H_bar_apo - to_rad(30.)) + 0.24 * cos(2.0 * H_bar_apo) + 0.32 * cos(3.0 * H_bar_apo + to_rad(6.0)) - 0.2 * cos(4.0 * H_bar_apo - to_rad(63.0));
    double sC = 1.0 + 0.045 * C_bar_apo;
    double sH = 1.0 + 0.015 * C_bar_apo * T;
    double sL = 1.0 + ((0.015 * pow(l_bar_apo - 50.0, 2.0)) / sqrt(20.0 + pow(l_bar_apo - 50.0, 2.0)));
    double RT = -2.0 * G * sin(to_rad(60.0) * exp(-pow((H_bar_apo - to_rad(275.0)) / to_rad(25.0), 2.0)));
    double res = (pow(delta_L_apo / (kL * sL), 2.0) + pow(delta_C_apo / (kC * sC), 2.0) + pow(delta_H_apo / (kH * sH), 2.0) + RT * (delta_C_apo / (kC * sC)) * (delta_H_apo / (kH * sH)));
    //return  sqrt(res);
    return res > 0 ? sqrt(res) : 0;
}


inline void invert_color(cv::Vec3b& point) {
    point[0] = 255 - point[0];
    point[1] = 255 - point[1];
    point[2] = 255 - point[2];
}

void flip_image(cv::Mat& image) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols / 2; j++) {
            std::swap(image.at<cv::Vec3b>(i, j), image.at<cv::Vec3b>(i, image.cols - j - 1));
        }
    }
}

void draw_capture_box0(cv::Mat& image) {
    const int row = image.rows;
    const int col = image.cols;

    int midx = col / 2;
    int midy = row / 2;
    int edge_size = ((row + col) / 2) / 2;

    int left = midx - edge_size / 2 * 6 / 4;
    int right = midx + edge_size / 2 * 6 / 4;
    int top = midy + edge_size / 2;
    int bottom = midy - edge_size / 2;
    for (int q = left; q <= right; q++) {
        invert_color(image.at<cv::Vec3b>(top, q));
        invert_color(image.at<cv::Vec3b>(bottom, q));
    }
    for (int q = bottom; q <= top; q++) {
        invert_color(image.at<cv::Vec3b>(q, left));
        invert_color(image.at<cv::Vec3b>(q, right));
    }

    int segment = edge_size / 4;
    for (int i = bottom + segment / 2; i < top; i += segment) {
        for (int j = right - segment / 2; j > left; j -= segment) {
            image.at<cv::Vec3b>(i, j) = { 0,255,0 };
            if (i == bottom + segment / 2 && j == right - segment / 2)
                image.at<cv::Vec3b>(i, j) = { 255,255,255 };
        }
    }
}

void draw_capture_box(cv::Mat& image) {
    const int row = image.rows;
    const int col = image.cols;

    int midx = col / 2;
    int midy = row / 2;
    int edge_size = ((row + col) / 2) / 4;

    int left = midx - edge_size / 2;
    int right = midx + edge_size / 2;
    int top = midy + edge_size / 2;
    int bottom = midy - edge_size / 2;
    for (int q = left; q <= right; q++) {
        invert_color(image.at<cv::Vec3b>(top, q));
        invert_color(image.at<cv::Vec3b>(bottom, q));
    }
    for (int q = bottom; q <= top; q++) {
        invert_color(image.at<cv::Vec3b>(q, left));
        invert_color(image.at<cv::Vec3b>(q, right));
    }
}

std::vector<cv::Vec3b> get_color_calibrator(cv::Mat& image) {
    std::vector<cv::Vec3b> result;
    const int row = image.rows;
    const int col = image.cols;

    int midx = col / 2;
    int midy = row / 2;
    int edge_size = ((row + col) / 2) / 2;

    int left = midx - edge_size / 2 * 6 / 4;
    int right = midx + edge_size / 2 * 6 / 4;
    int top = midy + edge_size / 2;
    int bottom = midy - edge_size / 2;

    int segment = edge_size / 4;
    cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
    for (int i = bottom + segment / 2; i < top; i += segment) {
        for (int j = left + segment / 2; j < right; j += segment) {
            double get_color[3] = { 0,0,0 };
            int pixel_count = 0;
            for (int k = i - 20; k <= i + 20; k++) {
                for (int w = j - 20; w <= j + 20; w++) {
                    get_color[0] += image.at<cv::Vec3b>(k, w)[0];
                    get_color[1] += image.at<cv::Vec3b>(k, w)[1];
                    get_color[2] += image.at<cv::Vec3b>(k, w)[2];
                    ++pixel_count;
                }
            }
            cv::Vec3b return_color;
            return_color[0] = get_color[0] / pixel_count;
            return_color[1] = get_color[1] / pixel_count;
            return_color[2] = get_color[2] / pixel_count;
            result.push_back(return_color);
        }
    }
    cv::cvtColor(image, image, cv::COLOR_Lab2BGR);
    return result;
}

cv::Vec3b get_color_capture_box(cv::Mat& image) {
    const int row = image.rows;
    const int col = image.cols;

    int midx = col / 2;
    int midy = row / 2;
    int edge_size = ((row + col) / 2) / 4;

    int left = midx - edge_size / 2;
    int right = midx + edge_size / 2;
    int top = midy + edge_size / 2;
    int bottom = midy - edge_size / 2;

    double get_color[3] = { 0,0,0 };
    int pixel_count = 0;
    for (int q = left; q <= right; q++) {
        for (int w = bottom; w <= top; w++) {
            get_color[0] += image.at<cv::Vec3b>(q, w)[0];
            get_color[1] += image.at<cv::Vec3b>(q, w)[1];
            get_color[2] += image.at<cv::Vec3b>(q, w)[2];
            ++pixel_count;
        }
    }
    cv::Vec3b return_color;
    return_color[0] = get_color[0] / pixel_count;
    return_color[1] = get_color[1] / pixel_count;
    return_color[2] = get_color[2] / pixel_count;
    return return_color;
}

bool is_command(const std::string& comparator, const std::string& str, int& pointer) {
    int _pointer = pointer;
    if (str.size() - pointer < comparator.size())
        return false;
    for (int i = 0; i < comparator.size(); i++) {
        if (str[_pointer] != comparator[i])
            return false;
        ++_pointer;
    }
    if (_pointer < str.size() && str[_pointer] != ' ')
        return false;
    pointer = _pointer + 1;
    return true;
}

void get_frame() {
    cv::Mat _frame;
    camera >> _frame;
    camera >> pure_frame;
    flip_image(_frame);
    tune(_frame);
    frame = _frame.clone();
}

void execute_command(const std::string& str) {
    int pointer = 0;
    if (is_command("capture", str, pointer)) {
        Ind.push_back({ "color123",get_color_capture_box(frame) });
        std::cout << "the color has success fully captured\n";
    }
    else if (is_command("save", str, pointer)) {
        put_indicator("indecator.txt", Ind);
        std::cout << "the indecator vector has successfully saved\n";
    }
    else if (is_command("tune", str, pointer)) {
        std::vector<cv::Vec3b> get_color = get_color_calibrator(pure_frame);
        init(color_base, get_color);
        std::cout << "tune successfully\n";
    }
    else {
        std::cout << "unknown command\n";
    }
}

void get_command() {
    get_command_terminate = true;
    while (get_command_terminate) {
        std::string command;
        std::getline(std::cin, command);
        execute_command(command);
    }
}

void show_camera() {
    while (true) {
        waiting_count++;
        cv::Mat get_frame = frame.clone();
        waiting_count--;

        if (get_frame.rows == 0)
            continue;
        draw_capture_box(get_frame);
        draw_capture_box0(get_frame);
        while (!show_camera_wait) {

        }
        get_camera = get_frame;
        show_camera_wait = false;
        cv::waitKey(1);
    }
}

void show_capture_color() {
    while (true) {
        waiting_count++;
        cv::Mat output_color = frame.clone();
        waiting_count--;

        if (output_color.rows == 0)
            continue;
        cv::Vec3b get_color = get_color_capture_box(output_color);
        for (int i = 0; i < output_color.rows; i++) {
            for (int j = 0; j < output_color.cols; j++) {
                output_color.at<cv::Vec3b>(i, j) = get_color;
            }
        }
        while (!show_color_capture_wait) {

        }
        get_color_capture = output_color;
        show_color_capture_wait = false;
        cv::waitKey(1);
    }
}

void show_color_detected() {
    while (true) {
        waiting_count++;
        cv::Mat color_frame = frame.clone();
        waiting_count--;

        if (color_frame.rows == 0)
            continue;

        cv::cvtColor(color_frame, color_frame, cv::COLOR_BGR2Lab);
        for (int i = 0; i < color_frame.rows; i += 5) {
            for (int j = 0; j < color_frame.cols; j += 5) {
                double min = 100000000;
                cv::Vec3b min_color;
                cv::Vec3b color_mean = { 0,0,0 };
                int count0 = 0;
                double a = 0, b = 0, c = 0;
                for (int k = 0; k < 5; k++)
                    for (int w = 0; w < 5; w++) {
                        a += color_frame.at<cv::Vec3b>(i + k, j + w)[0];
                        b += color_frame.at<cv::Vec3b>(i + k, j + w)[1];
                        c += color_frame.at<cv::Vec3b>(i + k, j + w)[2];
                        count0++;
                    }
                a /= count0;
                b /= count0;
                c /= count0;
                color_mean[0] = a;
                color_mean[1] = b;
                color_mean[2] = c;

                for (int k = 0; k < Ind.size(); k++) {
                    double distance = get_distance(Ind[k].color, color_mean);
                    if (distance < min) {
                        min = distance;
                        min_color = Ind[k].color;
                    }
                }
                for (int k = 0; k < 5; k++)
                    for (int w = 0; w < 5; w++)
                        color_frame.at<cv::Vec3b>(i + k, j + w) = min_color;
            }
        }
        cv::cvtColor(color_frame, color_frame, cv::COLOR_Lab2BGR);
        while (!show_color_detected_wait) {

        }
        get_color_detected = color_frame;
        show_color_detected_wait = false;
        cv::waitKey(1);
    }
}