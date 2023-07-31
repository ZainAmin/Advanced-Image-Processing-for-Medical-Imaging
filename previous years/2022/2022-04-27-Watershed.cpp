// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/coins2.jpg");

	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	cv::Mat binarized;
	cv::threshold(img_gray, binarized, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	cv::morphologyEx(binarized, binarized, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
	aia::imshow("Binarized", binarized, true, 1.0);

	cv::Mat dist_transform;
	cv::distanceTransform(binarized, dist_transform, cv::DIST_L2, 3);
	cv::normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);
	dist_transform.convertTo(dist_transform, CV_8U);
	aia::imshow("Distance transform", dist_transform, true, 1.0);

	cv::Mat dist_transform_bin;
	cv::threshold(dist_transform, dist_transform_bin, 0.6*255, 255, cv::THRESH_BINARY);
	aia::imshow("Distance transform thresholded", dist_transform_bin, true, 1.0);

	std::vector <std::vector <cv::Point> > objects;
	cv::findContours(dist_transform_bin, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	// internal markers
	cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));
	for (int i = 0; i < objects.size(); i++)
		cv::drawContours(markers, objects, i, cv::Scalar(i + 1), cv::FILLED);
	
	// external markers
	cv::Mat external_marker_mask;
	cv::erode(255 - binarized, external_marker_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	markers.setTo(cv::Scalar(objects.size()), external_marker_mask);

	// just for visualization purporses
	cv::Mat markers_vis;
	cv::normalize(markers, markers_vis, 0, 255, cv::NORM_MINMAX);
	markers_vis.convertTo(markers_vis, CV_8U);
	aia::imshow("Markers image", markers_vis, true, 1.0);

	cv::watershed(img, markers);

	// -1 = dams				--> *255 = -255 --> +255 = 0 --> CV_8U = 0
	// 0 = not present			--> ...			--> ...		 --> ...
	// [1, max index] = objects	--> >= 255		--> >= 255	 --> 255
	markers = markers * 255 + 255;
	markers.convertTo(markers, CV_8U);
	markers = 255 - markers;
	objects.clear();
	cv::findContours(markers, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	for (int i = 0; i < objects.size(); i++)
		cv::drawContours(img, objects, i, cv::Scalar(0, 0, 255), 2);
	aia::imshow("Segmented image", img, true, 1.0);

	return EXIT_SUCCESS;
}