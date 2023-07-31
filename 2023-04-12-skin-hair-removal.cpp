// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	
	cv::Mat img = cv::imread(
		std::string(EXAMPLE_IMAGES_PATH) + "/galaxy.jpg");

	aia::imshow("Original image", img, true, 0.5f);

	cv::Mat marker;
	cv::morphologyEx(img, marker, cv::MORPH_OPEN,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(41, 41)));
	aia::imshow("Marker", marker, true, 0.5f);

	cv::Mat mask = img;
	cv::Mat marker_prev;
	std::vector<cv::Mat> marker_channels(3);
	std::vector<cv::Mat> marker_prev_channels(3);
	do
	{
		marker_prev = marker.clone();

		cv::split(marker_prev, marker_prev_channels);

		cv::morphologyEx(marker, marker, cv::MORPH_DILATE,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

		cv::min(marker, mask, marker);

		cv::split(marker, marker_channels);

		aia::imshow("Reconstruction in progress", marker, false, 0.5f);
		cv::waitKey(10);

	} while (cv::countNonZero(marker_channels[0] - marker_prev_channels[0]) ||
		     cv::countNonZero(marker_channels[1] - marker_prev_channels[1]) ||
		     cv::countNonZero(marker_channels[2] - marker_prev_channels[2]));

	aia::imshow("Recontruction result", marker, true, 0.5f);
	aia::imshow("Stars result", img-marker, true, 0.5f);

	return EXIT_SUCCESS;
}

