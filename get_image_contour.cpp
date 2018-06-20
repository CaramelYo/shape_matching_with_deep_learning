#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

void get_image_contour(fs::path img_path, double threshold_0 = 50., double threshold_1 = 100., bool is_inverted = true);

// global variables

// data
fs::path data_dir = "data";
fs::path img_dir_path = data_dir / "image";
fs::path contour_img_dir_path = data_dir / "contour_img";
fs::path contour_data_dir_path = data_dir / "contour_data";
fs::path pre_result_img_dir_path = data_dir / "preprocessing_image";

fs::path fe_dir = "feature_extraction";
fs::path fe_pred_dir_path = data_dir / fe_dir / "pred";
fs::path fe_test_dir_path = data_dir / fe_dir / "test";
fs::path fe_pred_img_dir_path = data_dir / fe_dir / "pred_img";
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
                get_image_contour(img_path);
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

                // turn the contour_data into contour_img
                Mat pred_img = Mat::zeros(resized_width, resized_height, CV_8UC1);
                uint8_t *pred_img_data = (uint8_t *)pred_img.data;
                int pred_img_step = pred_img.step1(), pred_img_ch = pred_img.channels();
                uint8_t color = 255;

                for (size_t i = 0; i < contour.size(); ++i)
                {
                    // contour image
                    Point2i p = contour[i];
                    int index = p.y * pred_img_step + p.x * pred_img_ch;

                    pred_img_data[index] = color;
                }

                fs::path contour_stem = contour_data_path.stem();
                // combine the pred_img with the contour_img(ground truth)
                Mat contour_img = imread((contour_img_dir_path / contour_stem + ".png").c_str(), IMREAD_GRAYSCALE);

                if (is_inverted)
                {
                    threshold(pred_img, pred_img, 128., 255, THRESH_BINARY_INV);
                    threshold(contour_img, contour_img, 128., 255, THRESH_BINARY_INV);
                }

                imwrite((fe_pred_img_dir_path / contour_stem + ".png").c_str(), pred_img);

                Mat mixed_img;
                hconcat(contour_img, pred_img, mixed_img);

                imwrite((fe_mixed_img_dir_path / contour_stem + ".png").c_str(), mixed_img);
            }
        }
    }

    return 0;
}

void get_image_contour(fs::path img_path, double threshold_0, double threshold_1, bool is_inverted)
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
    Mat contour_img = Mat::zeros(padding_img.rows, padding_img.cols, CV_8UC1);
    uint8_t *contour_img_data = (uint8_t *)contour_img.data;
    int contour_img_step = contour_img.step1(), contour_img_ch = contour_img.channels();
    uint8_t color = 255;

    ofstream contour_data_file((contour_data_dir_path / to_string(img_counter)) + ".txt");

    for (size_t i = 0; i < max_contour.size(); ++i)
    {
        // contour image
        Point2i p = max_contour[i];
        int index = p.y * contour_img_step + p.x * contour_img_ch;

        contour_img_data[index] = color;

        // contour data
        contour_data_file << (float)p.x / resized_width << " " << (float)p.y / resized_height << "\n";
    }

    if (is_inverted)
    {
        threshold(canny_img, canny_img, 128., 255, THRESH_BINARY_INV);
        threshold(contour_img, contour_img, 128., 255, THRESH_BINARY_INV);
    }

    cvtColor(canny_img, canny_img, COLOR_GRAY2BGR);
    cvtColor(contour_img, contour_img, COLOR_GRAY2BGR);

    Mat pre_result_img;
    hconcat(padding_img, canny_img, pre_result_img);
    hconcat(pre_result_img, contour_img, pre_result_img);

    imwrite((contour_img_dir_path / to_string(img_counter) += img_path.extension()).c_str(), contour_img);
    imwrite((pre_result_img_dir_path / to_string(img_counter++) += img_path.extension()).c_str(), pre_result_img);

    contour_data_file.close();

    return;
}