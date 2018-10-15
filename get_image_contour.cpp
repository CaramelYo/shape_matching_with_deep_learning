#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>
#include <sstream>
#include <exception>

#include "mat_image.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

mat_image output_contour_image(const Size &img_size, const vector<Point2i> &contour, const uint8_t &color = 255);
mat_image output_contour_order_image(const Size &img_size, const vector<Point2i> &contour, const uint8_t &b = 255, const uint8_t &g = 0, uint8_t r = 0);
void output_contour_data(const fs::path &dir_path, const int &img_counter, const vector<Point2i> &contour);
void get_contour_image(fs::path img_path, double threshold_0 = 50., double threshold_1 = 100., bool is_inverted = true);
void check_dir_exist(initializer_list<fs::path> dir_path_list);
vector<Point2i> get_contour_data_from_file(fs::path path, const char del = ' ');

// global variables

// data
fs::path data_dir = "data";
fs::path img_dir_path = data_dir / "image";
fs::path contour_img_dir_path = data_dir / "contour_img";
fs::path full_contour_img_dir_path = data_dir / "full_contour_img";
fs::path binary_contour_img_dir_path = data_dir / "binary_contour_img";
fs::path contour_order_img_dir_path = data_dir / "contour_order_img";
fs::path contour_data_dir_path = data_dir / "contour_data";
fs::path pre_result_img_dir_path = data_dir / "preprocessing_image";

fs::path fe_dir = data_dir / "feature_extraction";
fs::path fe_pred_dir_path = fe_dir / "pred";
fs::path fe_pred_img_dir_path = fe_dir / "pred_img";
fs::path fe_pred_order_img_dir_path = fe_dir / "pred_order_img";
fs::path fe_mixed_img_dir_path = fe_dir / "mixed_img";
fs::path fe_selected_contour_dir = fe_dir / "40_80" / "selected_40_80_contour";
fs::path fe_selected_pre_img_dir = fe_dir / "40_80" / "selected_40_80_preprocessing_image";
fs::path fe_selected_fixed_num_contour_dir = fe_dir / "40_80" / "selected_fixed_num_40_80_contour";
fs::path fe_selected_fixed_num_pre_img_dir = fe_dir / "40_80" / "selected_fixed_num_40_80_preprocessing_image";

int img_counter = 0;

int resized_width = 512, resized_height = 512, padding_len = 5;
int no_padding_resized_width = resized_width - 2 * padding_len, no_padding_resized_height = resized_height - 2 * padding_len;

int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        string str_argv_1 = string(argv[1]);

        if (str_argv_1 == "create_contour_data")
        {
            // check whether those directories of result exist
            check_dir_exist({contour_img_dir_path,
                             full_contour_img_dir_path,
                             binary_contour_img_dir_path,
                             contour_order_img_dir_path,
                             contour_data_dir_path,
                             pre_result_img_dir_path});

            // read img name from dir
            img_counter = 0;

            for (fs::path img_path : fs::directory_iterator(img_dir_path))
                get_contour_image(img_path);
        }
        else if (str_argv_1 == "create_contour_img")
        {
            // check whether those directories of result exist
            check_dir_exist({fe_pred_img_dir_path,
                             fe_pred_order_img_dir_path,
                             fe_mixed_img_dir_path});

            bool is_inverted = true;

            for (fs::path contour_data_path : fs::directory_iterator(fe_pred_dir_path))
            {
                vector<Point2i> contour = get_contour_data_from_file(contour_data_path);

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
        else if (str_argv_1 == "fixed_n_contour_data")
        {
            check_dir_exist({fe_selected_fixed_num_contour_dir,
                             fe_selected_fixed_num_pre_img_dir});

            bool is_inverted = true;

            // "- 1" is used to ensure the length of new_contour_data is n(1500)
            // because I will append the first contour point to the last line of new_contour_data
            int fixed_n = 1500 - 1;
            // preserve the contour points in corners
            double point_length = 3.;

            for (fs::path contour_data_path : fs::directory_iterator(fe_selected_contour_dir))
            {
                vector<Point2i> contour = get_contour_data_from_file(contour_data_path);

                if (contour.size() < fixed_n)
                {
                    cerr << "contour size " << contour.size() << " is smaller than fixed_n " << fixed_n << endl;
                    exit(0);
                }

                // calculate the total "length" of contour (pdf)
                vector<double> contour_lengths;
                double total_contour_length = 0.;

                total_contour_length += point_length;
                contour_lengths.push_back(total_contour_length);

                for (size_t i = 0; i < contour.size() - 1; ++i)
                {
                    double d = norm(contour[i + 1] - contour[i]);

                    total_contour_length += d + point_length;
                    contour_lengths.push_back(total_contour_length);
                }

                // the length between the first and last points of contour
                total_contour_length += norm(contour[0] - contour[contour.size() - 1]) + point_length;
                contour_lengths.push_back(total_contour_length);

                double step = total_contour_length / fixed_n;
                size_t contour_lengths_size = contour_lengths.size(), contour_size = contour.size();

                assert(contour_lengths_size - 1 == contour_size);

                // create a new(fixed number) contour
                vector<Point2i> new_contour;
                double distance = 0.;
                int contour_lengths_i = 0;

                for (int i = 0; i < fixed_n; ++i)
                {
                    distance += step;

                    while (contour_lengths[contour_lengths_i] < distance && contour_lengths_i < contour_lengths_size)
                        ++contour_lengths_i;

                    Point2i p;

                    contour_lengths_i < contour_lengths_size ? contour_lengths_i : --contour_lengths_i;

                    double length = contour_lengths[contour_lengths_i], t = length - distance;

                    if (t < point_length)
                    {
                        // on the point
                        p = contour[contour_lengths_i < contour_size ? contour_lengths_i : contour_size - 1];
                    }
                    else
                    {
                        // on the edge
                        Point2i p0 = contour[contour_lengths_i - 1], p1;

                        // contour_size == contour_lengths_size - 1
                        p1 = contour_lengths_i < contour_size ? contour[contour_lengths_i] : contour[0];

                        Point2i dv = p0 - p1;
                        double d = norm(dv);
                        t -= point_length;

                        p = p1 + t / d * dv;
                    }

                    new_contour.push_back(p);
                }

                fs::path contour_stem = contour_data_path.stem();

                output_contour_data(fe_selected_fixed_num_contour_dir, stoi(contour_stem), new_contour);

                mat_image new_contour_img = output_contour_image(Size(resized_width, resized_height), new_contour);

                if (is_inverted)
                    threshold(new_contour_img.img, new_contour_img.img, 128., 255, THRESH_BINARY_INV);

                cvtColor(new_contour_img.img, new_contour_img.img, COLOR_GRAY2BGR);

                Mat pre_result_img = imread((fe_selected_pre_img_dir / contour_stem += ".png").c_str(), IMREAD_COLOR);
                hconcat(pre_result_img, new_contour_img.img, pre_result_img);

                imwrite((fe_selected_fixed_num_pre_img_dir / contour_stem += ".png").c_str(), pre_result_img);
            }
        }
        else
        {
            cerr << "unexpected sys argv[1] = " << str_argv_1 << endl;
            exit(0);
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
    ofstream file((dir_path / to_string(img_counter)) + ".txt");

    for (size_t i = 0; i < contour.size(); ++i)
    {
        Point2i p = contour[i];
        file << (float)p.x / resized_width << " " << (float)p.y / resized_height << "\n";
    }

    // output the first point at the last line of the data to preserve the continuity between the first and last points
    Point2i p = contour[0];
    file << (float)p.x / resized_width << " " << (float)p.y / resized_height << "\n";

    file.close();

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
    fs::path img_extension = img_path.extension();

    mat_image contour_img = output_contour_image(padding_img.size(), max_contour);
    mat_image binary_contour_img = output_contour_image(padding_img.size(), max_contour, 1);

    imwrite((full_contour_img_dir_path / to_string(img_counter) += img_extension).c_str(), contour_img.img);

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

    imwrite((contour_img_dir_path / to_string(img_counter) += img_extension).c_str(), contour_img.img);
    imwrite((binary_contour_img_dir_path / to_string(img_counter) += img_extension).c_str(), binary_contour_img.img);
    imwrite((contour_order_img_dir_path / to_string(img_counter) += img_extension).c_str(), contour_order_img.img);
    imwrite((pre_result_img_dir_path / to_string(img_counter++) += img_extension).c_str(), pre_result_img);

    return;
}

void check_dir_exist(initializer_list<fs::path> dir_path_list)
{
    for (fs::path dir_path : dir_path_list)
    {
        if (!fs::is_directory(dir_path))
        {
            fs::create_directory(dir_path);
            cout << "create the dir = " << dir_path << endl;
        }
    }
}

vector<Point2i> get_contour_data_from_file(fs::path path, const char del)
{
    // read the contour_data
    ifstream file(path.c_str());
    vector<Point2i> contour;

    string line;
    while (getline(file, line))
    {
        // split the line with del
        vector<string> elems;
        stringstream s_line(line);
        string elem;
        while (getline(s_line, elem, del))
            elems.push_back(elem);

        assert(elems.size() == 2);

        contour.push_back(Point2i(stof(elems[0]) * resized_width, stof(elems[1]) * resized_height));
    }

    return contour;
}