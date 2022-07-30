#include "Header.h"
#include "Function.h"
#include "Variable.h"
#include "linearzation_Function.h"


int main(int, char**) {
    load_indicator("indecator.txt",Ind);
    load_indicator("color_calibrator.txt", color_base);
    cv::namedWindow("Webcam", 1080);
    cv::namedWindow("Color_captured", 1080);
    cv::namedWindow("Color close", 1080);

    std::vector<LayerId> model_CCM = { { Layer::DENSE, 3, "act:linear dact:dlinear" } ,{ Layer::DENSE, 3 } };
    CCM.reconstruct(model_CCM,mean_squre_loss_func,dmean_squre_loss_func);
    CCM.rand_weight({ {0.25,0.35} });
    CCM.rand_bias({ {0.0,0.0} });
    CCM.set_all_learning_rate(3e-9);

    std::vector<LayerId> model_poly_trans = { { Layer::DENSE , 3, "act:linear dact:dlinear"} ,{ Layer::DENSE, 1} };

    red_poly_trans.reconstruct(model_poly_trans, mean_squre_loss_func, dmean_squre_loss_func);
    red_poly_trans.rand_weight({ {0,1} });
    red_poly_trans.rand_bias({ { 0.0,0.0 } });
    red_poly_trans.set_all_learning_rate(1e-13);

    green_poly_trans.reconstruct(model_poly_trans, mean_squre_loss_func, dmean_squre_loss_func);
    green_poly_trans.rand_weight({ {0,1} });
    green_poly_trans.rand_bias({ { 0.0,0.0 } });
    green_poly_trans.set_all_learning_rate(1e-13);

    blue_poly_trans.reconstruct(model_poly_trans, mean_squre_loss_func, dmean_squre_loss_func);
    blue_poly_trans.rand_weight({ {0,1} });
    blue_poly_trans.rand_bias({ { 0.0,0.0 } });
    blue_poly_trans.set_all_learning_rate(1e-13);


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