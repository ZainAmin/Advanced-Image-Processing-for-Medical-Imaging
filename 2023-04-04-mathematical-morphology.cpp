// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.png",
		cv::IMREAD_GRAYSCALE);

	std::vector<int> hist = ucas::histogram(img);

	int T_triangle = ucas::getTriangleAutoThreshold(hist);
	cv::Mat img_thresholded;
	cv::threshold(img, img_thresholded, T_triangle, 255, cv::THRESH_BINARY);
	cv::imshow("Triangle thresholding", img_thresholded);

	cv::morphologyEx(img_thresholded, img_thresholded, cv::MORPH_OPEN,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

	// uncomment this to check how contours can be extracted with MM
	/*cv::Mat img_eroded;
	cv::morphologyEx(img_thresholded, img_eroded, cv::MORPH_ERODE,
		cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	aia::imshow("Contour extraction with erosion", img_thresholded - img_eroded);*/

	std::vector< std::vector<cv::Point> > objects;
	cv::findContours(img_thresholded, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	printf("No. of objects = %d\n", objects.size());
	cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	cv::drawContours(img, objects, -1, cv::Scalar(0, 255, 255), 2);
	aia::imshow("Extracted objects", img);

	return EXIT_SUCCESS;
}