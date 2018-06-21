#ifndef MAT_IMAGE_H
#define MAT_IMAGE_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class mat_image
{
  public:
    mat_image(Mat img);

    int get_index(Point2i p);

    Mat img;
    int step, ch;
    uint8_t *data;
};

#endif