// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>


cv::Mat gradientImage(const cv::Mat & img)
{
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	//cv::GaussianBlur(img_gray, img_gray, cv::Size(0, 0), 5, 5);

	cv::Mat dx, dy;
	cv::Sobel(img_gray, dx, CV_32F, 1, 0);
	cv::Sobel(img_gray, dy, CV_32F, 0, 1);
	cv::Mat mag;
	cv::magnitude(dx, dy, mag);

	cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
	mag.convertTo(mag, CV_8U);

	return 3*mag;
}

int main()
{
	aia::processVideoStream("", gradientImage);

	return EXIT_SUCCESS;
}