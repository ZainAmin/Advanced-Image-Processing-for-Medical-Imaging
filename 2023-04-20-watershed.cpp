// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// we will need a simple struct to store rgb triplets
struct rgb {
	unsigned char r, g, b;
	rgb() : r(0), g(0), b(0) {}
	rgb(unsigned char _r, unsigned char _g, unsigned char _b) : r(_r), b(_b), g(_g) {}
};

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/coins2.jpg");

	aia::imshow("Original image", img);

	cv::Mat img_bin;
	cv::cvtColor(img, img_bin, cv::COLOR_BGR2GRAY);
	cv::threshold(img_bin, img_bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	aia::imshow("Binarized image", img_bin);

	cv::Mat img_bin_closed;
	cv::morphologyEx(img_bin, img_bin_closed, cv::MORPH_CLOSE,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
	aia::imshow("Closed binary image", img_bin_closed);

	cv::Mat dist_transform;
	cv::distanceTransform(img_bin_closed, dist_transform, cv::NORM_L1, 3);

	cv::Mat dist_transform_vis;
	cv::normalize(dist_transform, dist_transform_vis, 0, 255, cv::NORM_MINMAX);
	dist_transform_vis.convertTo(dist_transform_vis, CV_8U);
	aia::imshow("Distance transform", dist_transform_vis);

	cv::Mat dist_transform_bin;
	cv::threshold(dist_transform_vis, dist_transform_vis, 0.5*255, 255, cv::THRESH_BINARY);
	aia::imshow("Distance transform binarized", dist_transform_vis);

	cv::Mat img_overlaid_internal_markers = img.clone();
	img_overlaid_internal_markers.setTo(cv::Scalar(0, 0, 255), dist_transform_vis);
	aia::imshow("Internal markers overlaid", img_overlaid_internal_markers);

	cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));

	// internal markers
	std::vector< std::vector<cv::Point> > internal_markers_objs;
	cv::findContours(dist_transform_vis, internal_markers_objs, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	for (int k = 0; k < internal_markers_objs.size(); k++)
		cv::drawContours(markers, internal_markers_objs, k, cv::Scalar(k + 1), cv::FILLED);

	// external markers
	cv::Mat img_bin_dilated;
	cv::dilate(img_bin, img_bin_dilated,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(65, 65))); 
	aia::imshow("Dilated binarized image", img_bin_dilated);
	std::vector< std::vector<cv::Point> > external_markers_objs;
	cv::findContours(img_bin_dilated, external_markers_objs, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	cv::drawContours(markers, external_markers_objs, 0, cv::Scalar(internal_markers_objs.size() + 1), 2);

	cv::Mat markers_vis;
	cv::normalize(markers, markers_vis, 0, 255, cv::NORM_MINMAX);
	markers_vis.convertTo(markers_vis, CV_8U);
	aia::imshow("Markers image", markers_vis);


	// watershed
	cv::watershed(img, markers);


	// visualize dams
	cv::Mat dams = markers.clone();
	// dams is in [-1, internal_markers_objs.size() + 1] range
	// *255
	dams *= 255;
	// [-255, >255] range
	// +255
	dams += 255;
	// [0, >255] range
	// apply saturation (conversion to CV_8U)
	dams.convertTo(dams, CV_8U),
	// [0, 255] binary image where 0 is for dams
	// invert
	dams = 255 - dams;
	// binary image where 255 is for dams
	cv::dilate(dams, dams, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
	cv::Mat dam_watershed_output = img.clone();
	dam_watershed_output.setTo(cv::Scalar(0, 0, 255), dams);
	aia::imshow("Final result (dams)", dam_watershed_output);

	// visualize markers
	markers += 1;
	// we want to display the segmented regions with distinct colors
	// an easy (but not 100% accurate) way is to assign each object label a random color
	cv::Mat colored_watershed_output = img.clone();
	std::map <int, rgb> object2colors;
	for (int i = 1; i <= internal_markers_objs.size(); i++)
		object2colors[i] = rgb(rand() % 256, rand() % 256, rand() % 256);
	for (int y = 0; y < colored_watershed_output.rows; y++)
	{
		unsigned char* ythSegRow = colored_watershed_output.ptr<unsigned char>(y);
		int* ythWatRow = markers.ptr<int>(y);

		for (int x = 0; x < colored_watershed_output.cols; x++)
		{
			ythSegRow[3 * x + 0] = object2colors[ythWatRow[x]].b;
			ythSegRow[3 * x + 1] = object2colors[ythWatRow[x]].g;
			ythSegRow[3 * x + 2] = object2colors[ythWatRow[x]].r;
		}
	}
	aia::imshow("Final result (regions)", colored_watershed_output);

	return EXIT_SUCCESS;
}


