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

	std::vector< std::vector<cv::Point> > objects;
	cv::findContours(img_thresholded, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// filter by something
	objects.erase(std::remove_if(objects.begin(), objects.end(), 
		[](const std::vector<cv::Point>& object)
		{ 
			return cv::contourArea(object) < 100;
		}), objects.end());

	// filter by something else


	printf("No. of objects = %d\n", objects.size());
	cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	cv::drawContours(img, objects, -1, cv::Scalar(0, 255, 255), 2);
	aia::imshow("Extracted objects", img);

	return EXIT_SUCCESS;
}