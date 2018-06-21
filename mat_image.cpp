#include "mat_image.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

mat_image::mat_image(Mat img)
{
    this->img = img;
    this->step = img.step1();
    this->ch = img.channels();
    this->data = (uint8_t *)img.data;
}

int mat_image::get_index(Point2i p)
{
    return p.y * this->step + p.x * this->ch;
}