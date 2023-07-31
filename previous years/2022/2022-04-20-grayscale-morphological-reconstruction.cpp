// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int myCountNonZero(cv::Mat multichan_img)
{
	cv::Mat channels[3];
	cv::split(multichan_img, channels);
	return cv::countNonZero(channels[0]) + cv::countNonZero(channels[1]) + cv::countNonZero(channels[2]);
}

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/galaxy.jpg");

	cv::Mat marker;
	cv::morphologyEx(img, marker, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(37, 37)));
	aia::imshow("Marker", marker, true, 0.7);

	cv::Mat marker_cur = marker;
	cv::Mat marker_prev;
	cv::Mat mask = img;
	int iteration = 0;
	do
	{
		marker_prev = marker_cur.clone();

		cv::dilate(marker_cur, marker_cur, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
		marker_cur = cv::min(marker_cur, mask);

		aia::imshow("Reconstruction in progress", marker_cur, false, 0.7);
		cv::waitKey(100);

		printf("iteration = %d\n", ++iteration);

	} while (myCountNonZero(marker_cur - marker_prev));

	aia::imshow("Reconstructed image", marker_cur, true, 0.7);
	aia::imshow("Stars image", mask - marker_cur, 0.7);

	return EXIT_SUCCESS;
}