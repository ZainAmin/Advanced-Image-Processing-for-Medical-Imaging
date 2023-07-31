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
	/*cv::Mat img(600, 600, CV_8U, cv::Scalar(0));
	cv::rectangle(img, cv::Rect(200, 200, 300, 200), cv::Scalar(255));
	aia::imshow("Input image", img);*/

	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/text.png", cv::IMREAD_GRAYSCALE);

	std::vector<int> hist = ucas::histogram(img);
	int T = ucas::getOtsuAutoThreshold(hist);
	cv::threshold(img, img, T, 255, cv::THRESH_BINARY_INV);

	aia::imshow("Binary image", img);
	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/text_binarized.png", img);

	cv::Mat marker(img.rows, img.cols, CV_8U, cv::Scalar(0));
	cv::Mat marker_mask(img.rows, img.cols, CV_8U, cv::Scalar(0));
	cv::rectangle(marker_mask, cv::Rect(0, 0, img.cols-1, img.rows-1), cv::Scalar(255));
	aia::imshow("Mask", marker_mask);
	img = 255 - img;
	img.copyTo(marker, marker_mask);
	aia::imshow("Marker", marker);
	
	cv::Mat mask = img;
	cv::Mat marker_prev;
	do
	{
		marker_prev = marker.clone();

		cv::morphologyEx(marker, marker, cv::MORPH_DILATE,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

		marker = marker & mask;

		cv::imshow("Reconstruction in progress", marker);
		cv::waitKey(10);

	} while (cv::countNonZero(marker-marker_prev));

	aia::imshow("Filling result", 255-marker);

	return EXIT_SUCCESS;
}
