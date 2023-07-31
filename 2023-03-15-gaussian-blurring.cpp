// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int sigma = 1;
std::string winname = "Gaussian smoothing";
cv::Mat img;

void GaussianBlurring(int pos, void* userdata)
{
	if (sigma < 1)
		return;

	cv::Mat out_img(img.rows, img.cols, CV_8U);

	cv::GaussianBlur(img, out_img, cv::Size(0, 0), sigma, sigma);

	cv::imshow(winname, out_img);
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/girl.png");

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Sigma", winname, &sigma, 100, GaussianBlurring);

	GaussianBlurring(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}