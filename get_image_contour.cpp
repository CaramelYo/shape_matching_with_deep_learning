#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

void get_image_contour(fs::path img_path, double threshold_0 = 50., double threshold_1 = 150.);

// global variables
fs::path data_dir = "data", img_dir = "images", resized_img_dir = "resized_img", canny_dir = "canny", contour_img_dir = "contour_images", contour_dir = "contours";
fs::path img_dir_path = data_dir / img_dir;
fs::path resized_img_dir_path = data_dir / resized_img_dir;
fs::path canny_dir_path = data_dir / canny_dir;
fs::path contour_img_dir_path = data_dir / contour_img_dir;
fs::path contour_dir_path = data_dir / contour_dir;

int padding_len = 5, resized_width = 1024 - padding_len, resized_height = 1024 - padding_len;

int main(int argc, char **argv)
{
    // check whether those directories of result exist
    if (!fs::is_directory(resized_img_dir_path))
    {
        fs::create_directory(resized_img_dir_path);
        cout << "create the directory => " << resized_img_dir_path << endl;
    }

    if (!fs::is_directory(canny_dir_path))
    {
        fs::create_directory(canny_dir_path);
        cout << "create the directory => " << canny_dir_path << endl;
    }

    if (!fs::is_directory(contour_img_dir_path))
    {
        fs::create_directory(contour_img_dir_path);
        cout << "create the directory => " << contour_img_dir_path << endl;
    }

    if (!fs::is_directory(contour_dir_path))
    {
        fs::create_directory(contour_dir_path);
        cout << "create the directory => " << contour_dir_path << endl;
    }

    // read img name from dir
    for (fs::path img_path : fs::directory_iterator(img_dir_path))
    {
        get_image_contour(img_path);
    }

    return 0;
}

void get_image_contour(fs::path img_path, double threshold_0, double threshold_1)
{
    Mat img = imread(img_path.c_str(), IMREAD_COLOR);

    if (!img.data)
    {
        cerr << "error in loading image" << endl;
        return;
    }

    // resize the img
    Mat resized_img;
    resize(img, resized_img, Size(resized_width, resized_height));
    imwrite((resized_img_dir_path / img_path.filename()).c_str(), resized_img);

    // pad the img
    Mat padding_img;
    Scalar border(255, 255, 255);
    copyMakeBorder(resized_img, padding_img, padding_len, padding_len, padding_len, padding_len, BORDER_CONSTANT, border);

    // canny first
    Mat blur_img;
    blur(padding_img, blur_img, Size(3, 3));

    Mat canny_img;
    Canny(blur_img, canny_img, threshold_0, threshold_1, 3);
    imwrite((canny_dir_path / img_path.filename()).c_str(), canny_img);

    // find the contours
    vector<vector<Point>> contours;
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
    Mat temp_img = Mat::zeros(padding_img.rows, padding_img.cols, CV_8UC1);
    uint8_t *temp_img_data = (uint8_t *)temp_img.data;
    int temp_img_step = temp_img.step1(), temp_img_ch = temp_img.channels();
    uint8_t color = 255;

    ofstream contour_file;
    contour_file.open((contour_dir_path / img_path.stem()) + ".txt");

    for (size_t i = 0; i < max_contour.size(); ++i)
    {
        // contour image
        Point2i p = max_contour[i];
        int index = p.y * temp_img_step + p.x * temp_img_ch;

        temp_img_data[index] = color;

        // contour data
        contour_file << (double)p.x / resized_width << " " << (double)p.y / resized_height << "\n";
    }

    imwrite((contour_img_dir_path / img_path.filename()).c_str(), temp_img);

    contour_file.close();

    return;
}