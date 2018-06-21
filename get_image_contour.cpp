#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>
#include <sstream>

#include "mat_image.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

mat_image output_contour_image(const Size &img_size, const vector<Point2i> &contour, const uint8_t &color = 255);
mat_image output_contour_order_image(const Size &img_size, const vector<Point2i> &contour, const uint8_t &b = 255, const uint8_t &g = 0, uint8_t r = 0);
void output_contour_data(const fs::path &dir_path, const int &img_counter, const vector<Point2i> &contour);
void get_contour_image(fs::path img_path, double threshold_0 = 40., double threshold_1 = 80., bool is_inverted = true);

// global variables

// data
fs::path data_dir = "data";
fs::path img_dir_path = data_dir / "image";
fs::path contour_img_dir_path = data_dir / "contour_img";
fs::path contour_order_img_dir_path = data_dir / "contour_order_img";
fs::path contour_data_dir_path = data_dir / "contour_data";
fs::path pre_result_img_dir_path = data_dir / "preprocessing_image";

fs::path fe_dir = "feature_extraction";
fs::path fe_pred_dir_path = data_dir / fe_dir / "pred";
fs::path fe_test_dir_path = data_dir / fe_dir / "test";
fs::path fe_pred_img_dir_path = data_dir / fe_dir / "pred_img";
fs::path fe_pred_order_img_dir_path = data_dir / fe_dir / "pred_order_img";
fs::path fe_mixed_img_dir_path = data_dir / fe_dir / "mixed_img";

int img_counter = 0;

int resized_width = 1024, resized_height = 1024, padding_len = 5;
int no_padding_resized_width = resized_width - 2 * padding_len, no_padding_resized_height = resized_height - 2 * padding_len;

int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        string str_argv_1 = string(argv[1]);

        if (str_argv_1 == "create_contour_data")
        {
            // check whether those directories of result exist
            if (!fs::is_directory(contour_img_dir_path))
            {
                fs::create_directory(contour_img_dir_path);
                cout << "create the dir = " << contour_img_dir_path << endl;
            }

            if (!fs::is_directory(contour_order_img_dir_path))
            {
                fs::create_directory(contour_order_img_dir_path);
                cout << "create the dir = " << contour_order_img_dir_path << endl;
            }

            if (!fs::is_directory(contour_data_dir_path))
            {
                fs::create_directory(contour_data_dir_path);
                cout << "create the dir = " << contour_data_dir_path << endl;
            }

            if (!fs::is_directory(pre_result_img_dir_path))
            {
                fs::create_directory(pre_result_img_dir_path);
                cout << "create the dir = " << pre_result_img_dir_path << endl;
            }

            // read img name from dir
            for (fs::path img_path : fs::directory_iterator(img_dir_path))
            {
                get_contour_image(img_path);
            }
        }
        else if (str_argv_1 == "create_contour_img")
        {
            // check whether those directories of result exist
            if (!fs::is_directory(fe_pred_img_dir_path))
            {
                fs::create_directory(fe_pred_img_dir_path);
                cout << "create the dir = " << fe_pred_img_dir_path << endl;
            }

            if (!fs::is_directory(fe_pred_order_img_dir_path))
            {
                fs::create_directory(fe_pred_order_img_dir_path);
                cout << "create the dir = " << fe_pred_order_img_dir_path << endl;
            }

            if (!fs::is_directory(fe_mixed_img_dir_path))
            {
                fs::create_directory(fe_mixed_img_dir_path);
                cout << "create the dir = " << fe_mixed_img_dir_path << endl;
            }

            bool is_inverted = true;

            for (fs::path contour_data_path : fs::directory_iterator(fe_pred_dir_path))
            {
                // read the contour_data
                ifstream contour_data_file(contour_data_path.c_str());

                vector<Point2i> contour;
                const char del = ' ';
                string line;
                while (getline(contour_data_file, line))
                {
                    // split the line with ' '
                    vector<string> elems;
                    stringstream s_line(line);
                    string elem;
                    while (getline(s_line, elem, del))
                    {
                        elems.push_back(elem);
                    }

                    assert(elems.size() == 2);

                    contour.push_back(Point2i(stof(elems[0]) * resized_width, stof(elems[1]) * resized_height));
                }

                mat_image pred_img = output_contour_image(Size(resized_width, resized_height), contour);

                fs::path contour_stem = contour_data_path.stem();
                // combine the pred_img with the contour_img(ground truth)
                Mat contour_img = imread((contour_img_dir_path / contour_stem + ".png").c_str(), IMREAD_GRAYSCALE);

                if (is_inverted)
                {
                    threshold(pred_img.img, pred_img.img, 128., 255, THRESH_BINARY_INV);
                    threshold(contour_img, contour_img, 128., 255, THRESH_BINARY_INV);
                }

                imwrite((fe_pred_img_dir_path / contour_stem + ".png").c_str(), pred_img.img);

                Mat mixed_img;
                hconcat(contour_img, pred_img.img, mixed_img);

                imwrite((fe_mixed_img_dir_path / contour_stem + ".png").c_str(), mixed_img);
            }
        }
    }

    return 0;
}

mat_image output_contour_image(const Size &img_size, const vector<Point2i> &contour, const uint8_t &color)
{
    mat_image contour_img(Mat::zeros(img_size, CV_8UC1));

    for (size_t i = 0; i < contour.size(); ++i)
    {
        Point2i p = contour[i];
        int index = contour_img.get_index(p);

        contour_img.data[index] = color;
    }

    return contour_img;
}

mat_image output_contour_order_image(const Size &img_size, const vector<Point2i> &contour, const uint8_t &b, const uint8_t &g, uint8_t r)
{
    // show the order of contour with gradient color
    mat_image contour_order_img(Mat::zeros(img_size, CV_8UC3));
    contour_order_img.img = Scalar(255, 255, 255);

    uint8_t colors[3] = {b, g, r};
    float color_r = 0., color_r_step = 255. / contour.size();

    for (size_t i = 0; i < contour.size(); ++i)
    {
        Point2i p = contour[i];
        int index = contour_order_img.get_index(p);

        for (unsigned j = 0; j < contour_order_img.ch; ++j)
        {
            contour_order_img.data[index + j] = colors[j];
        }

        // gradient color
        color_r += color_r_step;
        colors[2] = (uint8_t)color_r;
    }

    return contour_order_img;
}

void output_contour_data(const fs::path &dir_path, const int &img_counter, const vector<Point2i> &contour)
{
    ofstream contour_data((dir_path / to_string(img_counter)) + ".txt");

    for (size_t i = 0; i < contour.size(); ++i)
    {
        Point2i p = contour[i];

        // contour data
        contour_data << (float)p.x / resized_width << " " << (float)p.y / resized_height << "\n";
    }

    // output the first point at the last line of the data to preserve the continuity between the first and last points
    Point2i p = contour[0];
    contour_data << (float)p.x / resized_width << " " << (float)p.y / resized_height << "\n";

    contour_data.close();

    return;
}

void get_contour_image(fs::path img_path, double threshold_0, double threshold_1, bool is_inverted)
{
    Mat img = imread(img_path.c_str(), IMREAD_COLOR);

    if (!img.data)
    {
        cerr << "error in loading image" << endl;
        return;
    }

    // resize the img
    Mat resized_img;
    resize(img, resized_img, Size(no_padding_resized_width, no_padding_resized_height));

    // pad the img
    Mat padding_img;
    Scalar border(255, 255, 255);
    copyMakeBorder(resized_img, padding_img, padding_len, padding_len, padding_len, padding_len, BORDER_CONSTANT, border);

    // canny first
    Mat blur_img;
    blur(padding_img, blur_img, Size(3, 3));

    Mat canny_img;
    Canny(blur_img, canny_img, threshold_0, threshold_1, 3);
    // imwrite((canny_dir_path / img_path.filename()).c_str(), canny_img);

    // find the contours
    vector<vector<Point2i>> contours;
    findContours(canny_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // find the contour with max length
    size_t max_contour_index = 0, max_contour_length = 0;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        size_t length = contours[i].size();
        if (max_contour_length < length)
        {
            max_contour_length = length;
            max_contour_index = i;
        }
    }

    vector<Point> max_contour = contours[max_contour_index];

    // output the max_contour
    mat_image contour_img = output_contour_image(padding_img.size(), max_contour);
    mat_image contour_order_img = output_contour_order_image(padding_img.size(), max_contour);
    output_contour_data(contour_data_dir_path, img_counter, max_contour);

    if (is_inverted)
    {
        threshold(canny_img, canny_img, 128., 255, THRESH_BINARY_INV);
        threshold(contour_img.img, contour_img.img, 128., 255, THRESH_BINARY_INV);
    }

    cvtColor(canny_img, canny_img, COLOR_GRAY2BGR);
    cvtColor(contour_img.img, contour_img.img, COLOR_GRAY2BGR);

    Mat pre_result_img;
    hconcat(padding_img, canny_img, pre_result_img);
    hconcat(pre_result_img, contour_img.img, pre_result_img);

    imwrite((contour_img_dir_path / to_string(img_counter) += img_path.extension()).c_str(), contour_img.img);
    imwrite((contour_order_img_dir_path / to_string(img_counter) += img_path.extension()).c_str(), contour_order_img.img);
    imwrite((pre_result_img_dir_path / to_string(img_counter++) += img_path.extension()).c_str(), pre_result_img);

    return;
}