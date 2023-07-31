// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"


// GOAL: Marker-controlled Watershed segmentation
// example images used in this code are within the 'example_images' folder within the project source folder
// and/or can be downloaded from https://www.dropbox.com/sh/glo9si2fu1x3776/AADFn1o3nYsV2OFieW-KEXn7a?dl=1
int main() 
{
	try
	{
		// load "pills" image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pills.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");
		aia::imshow("Color image", img, true, 2.0f);
		
		// convert to grayscale
		cv::Mat img_gray;
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		aia::imshow("Grayscale Image", img_gray, true, 2.0f);

		// binarize image so we can manipulate objects
		cv::threshold(img_gray, img_gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		aia::imshow("Otsu", img_gray, true, 2.0f);

		// remove small objects (noise)
		cv::morphologyEx(img_gray, img_gray, CV_MOP_OPEN, cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3,3)));
		aia::imshow("Otsu + Opening", img_gray, true, 2.0f);

		// calculate distance transform
		cv::Mat dist;
		cv::distanceTransform(img_gray, dist, CV_DIST_L2, CV_DIST_MASK_3);

		// convert distance transform output (real-valued matrix) into a [0,255]-valued image
		// 'dist' values are in [min, max] --> let's find 'min' and 'max' !
		double min, max;
		cv::minMaxLoc(dist, &min, &max);
		dist = dist - min;					// shift values to the right --> values will be in [0, max-min]
		dist = dist * (255 / (max-min));	// rescale values            --> values will be in [0, 255]
		dist.convertTo(dist, CV_8U);
		aia::imshow("Otsu + Opening + DT", dist, true, 2.0f);

		// we want to build internal markers using the distance transform output
		// we need a 'reasonable' binarization to select only the most internal points
		cv::threshold(dist, dist, 180, 255, CV_THRESH_BINARY);
		aia::imshow("Otsu + Opening + DT + thresh", dist, true, 2.0f);

		// we are now ready to extract the internal markers = connected components from the previous step
		std::vector <std::vector <cv::Point> > internal_markers;
		cv::findContours(dist, internal_markers, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		// a marker image is a integer-valued matrix of object labels where
		// 0 = unknown labels
		// positive values = object values (e.g. object 1 --> label 'x', object 2 --> label 'y', etc.)
		cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));	// initialize all pixels with '0'

		// we have internal markers --> we can insert them into the 'markers' image
		for(int k=0; k<internal_markers.size(); k++)
			cv::drawContours(markers, internal_markers, k, cv::Scalar(k+1), CV_FILLED);
		aia::imshow("internal markers", markers, true, 2.0f);
		// ...this is an integer-valued image with values up to billions, it's not suited for visualization!
		// if we really want to visualize it, we have to rescale it
		// by construction, 'markers' has values in [0, internal_markers.size()]
		cv::Mat markers_vis = markers.clone();
		markers_vis = markers_vis * (255.0 / (internal_markers.size()));
		markers_vis.convertTo(markers_vis, CV_8U);
		aia::imshow("internal markers (for visualization)", markers_vis, true, 2.0f);

		// we build external markers by performing a (big) dilation of the previously generated binary image
		cv::dilate(img_gray, img_gray, cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(21,21)));
		aia::imshow("Otsu + Opening + Dilate", img_gray, true, 2.0f);

		// we are now ready to extract the external markers = connected components from the previous step
		std::vector <std::vector <cv::Point> > external_markers;
		cv::findContours(img_gray, external_markers, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		// we have external markers --> we can insert them into the 'markers' image
		cv::drawContours(markers, external_markers, 0, cv::Scalar(internal_markers.size()+1));	// we use a label we are sure we did not used before
		aia::imshow("all markers", markers, true, 2.0f);
		// ...this is an integer-valued image with values up to billions, it's not suited for visualization!
		// if we really want to visualize it, we have to rescale it
		// by construction, 'markers' has values in [0, internal_markers.size() + 1]
		markers_vis = markers.clone();
		markers_vis = markers_vis * (255.0f / (internal_markers.size()+1));
		markers_vis.convertTo(markers_vis, CV_8U);
		aia::imshow("all markers (for visualization)", markers_vis, true, 2.0f);

		// finally, we can perform the watershed
		cv::watershed(img, markers);
		//            /\
		//            || why on the original color image? Because OpenCV (and also MATLAB) implements the
		//               "Meyer, F. Color Image Segmentation, ICIP92, 1992" paper that can deal with colors


		// by construction, 'markers' has values in [-1, internal_markers.size() + 1]
		// where '-1' corresponds to the dams
		// thus, if we shift by 1 to the right, values will be in [0, internal_markers.size() + 2]
		markers_vis = markers.clone();
		markers_vis += 1;
		markers_vis = markers_vis * (255.0f / (internal_markers.size()+2));
		markers_vis.convertTo(markers_vis, CV_8U);
		aia::imshow("watershed result", markers_vis, true, 2.0f);

		// if we are only interested to contours to be overimposed on the original image,
		// we can do some simple reasoning on how to rescale the values of 'markers' in the 8-bit range [0, 255]
		// pixels whose value is -1   : dams  -------------------->  -1*255+255 = 0
		// pixels whose value is >= 0 : objects and background --->   x*255+255 = 255 (saturation)
		markers.convertTo(markers, CV_8U, 255, 255);
		// image is now 'all white' except for the dams ('black') --> we have a binary image!

		// we can now find the contours on the inverted binary image
		std::vector < std::vector <cv::Point> > segmented_objects;
		markers = 255-markers;
		cv::findContours(markers, segmented_objects, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

		// and overimpose them on the original image
		cv::drawContours(img, segmented_objects, -1, cv::Scalar(0,0,255), 2, CV_AA);
		aia::imshow("final result", img, true, 2.0f);

		return 1;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}