#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // path
    string data_dir = "data/", img_dir = "images/";
    string img_dir_path = string().append(data_dir).append(img_dir);

    // read img
    Mat img = imread(string().append(img_dir_path).append("sample_0.png"), IMREAD_COLOR);

    if (!img.data)
    {
        cerr << "error in loading image" << endl;
        return -1;
    }

    // pad the img
    Mat padding_img;
    Scalar border(255, 255, 255);
    int padding_len = 5;
    copyMakeBorder(img, padding_img, padding_len, padding_len, padding_len, padding_len, BORDER_CONSTANT, border);

    // canny first
    Mat blur_img;
    blur(padding_img, blur_img, Size(3, 3));

    double threshold_0 = 50., threshold_1 = 150.;

    Mat canny_img;
    Canny(blur_img, canny_img, threshold_0, threshold_1, 3);

    namedWindow("canny_img", WINDOW_AUTOSIZE);
    imshow("canny_img", canny_img);

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

    Mat temp_img = Mat::zeros(padding_img.rows, padding_img.cols, CV_8UC1);
    uint8_t *temp_img_data = (uint8_t *)temp_img.data;
    int temp_img_step = temp_img.step1(), temp_img_ch = temp_img.channels();

    uint8_t color = 255;

    for (size_t i = 0; i < max_contour.size(); ++i)
    {
        Point2i p = max_contour[i];
        int index = p.y * temp_img_step + p.x * temp_img_ch;

        temp_img_data[index] = color;
    }

    namedWindow("temp_img", WINDOW_AUTOSIZE);
    imshow("temp_img", temp_img);
    waitKey(0);

    return 0;
}