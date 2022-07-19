#include "opencv2/opencv.hpp"
#include "iostream"
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <atomic>
#define PI 3.14159265358979323846

bool get_command_terminate = true;
cv::VideoCapture camera(0);
cv::Mat frame;
cv::Mat pure_frame;
cv::Mat get_camera;
cv::Mat get_color_capture;
cv::Mat get_color_detected;

std::atomic<int> waiting_count = 0;
bool show_camera_wait = true;
bool show_color_capture_wait = true;
bool show_color_detected_wait = true;
int tuner[3] = {0,0,0};

class indicator {
public:
    std::string name;
    cv::Vec3b color;
};

std::vector<indicator> Ind;
std::vector<indicator> color_base;

void load_indicator(const std::string& input_file_name,std::vector<indicator>& Ind) {
    std::ifstream input_file(input_file_name);
    while (!input_file.eof()) {
        indicator _Ind;
        std::getline(input_file, _Ind.name);
        int B, G, R;
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



double get_distance(const cv::Vec3b& p1,const cv::Vec3b& p2) { // LAB color space

    const double KL = 1;
    const double KC = 1;
    const double KH = 1;

    auto get_LCh = [](cv::Vec3b p) {
        cv::Vec3b result;
        result[0] = p[0]; // L
        result[1] = std::sqrt(std::pow(p[1], 2) + std::pow(p[2], 2));// C
        result[2] = std::atan2(p[2], p[1]) * double(180) / PI;
        return result;
    };

    cv::Vec3b p1LCh = get_LCh(p1);
    cv::Vec3b p2LCh = get_LCh(p2);

    double avrC = (p1LCh[1] + p2LCh[2]) / double(2);
    double G = 0.5 * (double(1) - std::sqrt(std::pow(avrC, 7) / (std::pow(avrC, 7) + std::pow(25, 7))));

    double a1_ = (1 + G) * p1[1];
    double a2_ = (1 + G) * p2[1];

    double C1_ = std::sqrt(std::pow(a1_, 2) + std::pow(p1[2], 2));
    double C2_ = std::sqrt(std::pow(a2_, 2) + std::pow(p2[2], 2));

    double h1_ = (p1[2] == a1_ && a1_ == 0) ? 0 : std::atan2(p1[2], a1_);
    double h2_ = (p2[2] == a2_ && a2_ == 0) ? 0 : std::atan2(p2[2], a2_);

    double dL = p2[0] - p1[0];
    double dC = C2_ - C1_;
    double dh = C1_ * C2_ == 0 ? 0 :
        std::abs(h2_ - h1_) <= 180 ? h2_ - h1_ :
        h2_ - h1_ > 180 ? h2_ - h1_ - 360 :
        h2_ - h1_ + 360;

    double dH = 2 * std::sqrt(C1_ * C2_) * std::sin(dh / 2);

    double avrL = (p1LCh[0] + p2LCh[1]) / 2;
    double avrC_ = (C1_ + C2_) / 2;
    double avrh = C1_ * C2_ == 0 ? h1_ + h2_ :
        std::abs(h1_ - h2_) <= 180 ? (h1_ + h2_) / 2 :
        h1_ + h2_ < 360 ? (h1_ + h2_ + 360) / 2 :
        (h1_ + h2_ - 360) / 2;

    double T = double(1)
        - 0.17 * std::cos(avrh - 30)
        + 0.24 * std::cos(2 * avrh)
        + 0.32 * std::cos(3 * avrh + 6)
        - 0.20 * std::cos(4 * avrh - 63);

    double dangle = 30 * std::exp(-(avrh - 275) / 25);
    double Rc = 2 * std::sqrt(std::pow(avrC_, 7) / (std::pow(avrC_, 7) + std::pow(25, 7)));

    double SL = double(1) + (0.015) * std::pow(avrL - 50, 2) / std::sqrt(20 + std::pow(avrL - 50, 2));
    double SC = double(1) + 0.045 * avrC_;
    double SH = double(1) + 0.015 * avrC_ * T;
    double RT = -std::sin(2 * dangle * PI / 180) * Rc;

    double result = std::sqrt(
        std::pow(dL / (KL * SL), 2)
        + std::pow(dC / (KC * SC), 2)
        + std::pow(dH / (KH * SH), 2)
        + RT * (dC / (KC * SC)) * (dH / (KH * SH)));
    return result;
}

inline void invert_color(cv::Vec3b& point) {
    point[0] = 255 - point[0];
    point[1] = 255 - point[1];
    point[2] = 255 - point[2];
}

void flip_image(cv::Mat& image) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols / 2; j++) {
            std::swap(image.at<cv::Vec3b>(i, j), image.at<cv::Vec3b>(i,image.cols - j - 1));
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
            image.at<cv::Vec3b>(i, j) = {0,255,0};
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
    for (int i = bottom + segment / 2; i < top; i += segment) {
        for (int j = left + segment / 2; j < right; j += segment) {
            result.push_back(image.at<cv::Vec3b>(i, j));
        }
    }
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

    double get_color[3] = {128,0,0};
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

bool is_command(const std::string& comparator,const std::string& str,int& pointer) {
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

void filter_image(cv::Mat& image) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < 3; k++) {
                int a = image.at<cv::Vec3b>(i, j)[k];
                a += tuner[k];
                if (a < 0) a = 0;
                if (a > 255)a = 255;
                image.at<cv::Vec3b>(i, j)[k] = a;
            }
        }
    }
}

void get_frame() {
    cv::Mat _frame;
    camera >> _frame;
    camera >> pure_frame;
    flip_image(_frame);
    filter_image(_frame);
    frame = _frame.clone();
}

void execute_command(const std::string& str) {
    int pointer = 0;
    if (is_command("capture", str, pointer)) {
        Ind.push_back({ "color123",get_color_capture_box(frame) });
        std::cout << "the color has success fully captured\n";
    }
    else if (is_command("save", str, pointer)) {
        put_indicator("indecator.txt",Ind);
        std::cout << "the indecator vector has successfully saved\n";
    }
    else if (is_command("tune", str, pointer)) {
        std::vector<cv::Vec3b> get_color = get_color_calibrator(pure_frame);
        int b = 0, g = 0, r = 0;
        for (int i = 0; i < get_color.size(); i++) {
            b += color_base[i].color[0] - get_color[i][0];
            g += color_base[i].color[1] - get_color[i][1];
            r += color_base[i].color[2] - get_color[i][2];
            std::cout << color_base[i].color[0] - get_color[i][0] << " " << color_base[i].color[1] - get_color[i][1] << " " << color_base[i].color[2] - get_color[i][2] << std::endl;
        }
        b /= (int)get_color.size();
        g /= (int)get_color.size();
        r /= (int)get_color.size();
        tuner[0] = b;
        tuner[1] = g;
        tuner[2] = r;
        std::cout << b << ' ' << g << " " << r << std::endl;
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
        for (int i = 0; i < color_frame.rows; i+=2) {
            for (int j = 0; j < color_frame.cols; j+=2) {
                double min = 100000000;
                cv::Vec3b min_color;
                for (int k = 0; k < Ind.size(); k++) {
                    double distance = get_distance(Ind[k].color, color_frame.at<cv::Vec3b>(i, j));
                    if (distance < min) {
                        min = distance;
                        min_color = Ind[k].color;
                    }
                }
                for (int k = 0; k < 2; k++) 
                    for(int w = 0; w < 2; w++)
                        color_frame.at<cv::Vec3b>(i + k, j + w) = min_color;
            }
        }
        while (!show_color_detected_wait) {

        }
        get_color_detected = color_frame;
        show_color_detected_wait = false;
        cv::waitKey(1);
    }
}


int main(int, char**) {
    load_indicator("indecator.txt",Ind);
    load_indicator("color_calibrator.txt", color_base);
    cv::namedWindow("Webcam", 1080);
    cv::namedWindow("Color_captured", 1080);
    cv::namedWindow("Color close", 1080);

    std::thread get_command_thread(get_command);
    std::thread show1(show_camera);
    std::thread show2(show_capture_color);
    std::thread show3(show_color_detected);


    while (true) {
        try {
            if (waiting_count == 0)
                get_frame();
            if (get_camera.rows != 0 && !show_camera_wait) {
                imshow("Webcam", get_camera);
                show_camera_wait = true;
            }
            if (get_color_capture.rows != 0 && !show_color_capture_wait) {
                imshow("Color_captured", get_color_capture);
                show_color_capture_wait = true;
            }
            if (get_color_detected.rows != 0 && !show_color_detected_wait) {
                imshow("Color close", get_color_detected);
                show_color_detected_wait = true;
            }
            cv::waitKey(1);
        }
        catch (int error) {

        }
    }
    return 0;
}