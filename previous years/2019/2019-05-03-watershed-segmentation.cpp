// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"


int main() 
{
	try
	{
		// load the image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pills.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");

		// show the image
		aia::imshow("Image", img, true, 2.0f);

		// grayscale conversion
		cv::Mat img_gray;
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

		// binarization
		cv::Mat binarized;
		cv::threshold(img_gray, binarized, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		aia::imshow("Binarized", binarized, true, 2.0f);

		// removing artifacts
		cv::morphologyEx(binarized, binarized, cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
		aia::imshow("Artificats removal", binarized, true, 2.0f);

		// distance transform
		cv::Mat img_distance;
		cv::distanceTransform(binarized, img_distance, CV_DIST_L1, 3);
		cv::normalize(img_distance, img_distance, 0, 255, cv::NORM_MINMAX);
		img_distance.convertTo(img_distance, CV_8U);
		aia::imshow("Distance transform", img_distance, true, 2.0f);

		// thresholded distance transform
		cv::threshold(img_distance, img_distance, 150, 255, CV_THRESH_BINARY);
		img_gray.setTo(cv::Scalar(0), img_distance);
		aia::imshow("Evaluation of internal markers", img_gray, true, 2.0f);

		// internal marker generation / labeling
		std::vector < std::vector <cv::Point> > contours;
		cv::findContours(img_distance, contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));
		for(int i=0; i<contours.size(); i++)
			cv::drawContours(markers, contours, i, cv::Scalar(i+1), CV_FILLED);
		
		// visualize internal marker
		cv::Mat markers_img;
		cv::normalize(markers, markers_img, 0, 255, cv::NORM_MINMAX);
		markers_img.convertTo(markers_img, CV_8U);
		aia::imshow("Internal markers image", markers_img, true, 2.0f);

		// generate external markers
		cv::Mat external_marker;
		cv::morphologyEx(binarized, external_marker, cv::MORPH_DILATE,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21,21)));
		int number_of_objects = contours.size();
		contours.clear();
		cv::findContours(external_marker, contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		cv::drawContours(img_gray, contours, 0, cv::Scalar(0));
		aia::imshow("Evaluation of external marker", img_gray, true, 2.0f);

		// include external marker into the markers matrix
		cv::drawContours(markers, contours, 0, cv::Scalar(number_of_objects+1));
		cv::normalize(markers, markers_img, 0, 255, cv::NORM_MINMAX);
		markers_img.convertTo(markers_img, CV_8U);
		aia::imshow("Internal + external markers image", markers_img, true, 2.0f);

		// watershed
		cv::watershed(img, markers);
		cv::Mat watershed_result;
		cv::normalize(markers, watershed_result, 0, 255, cv::NORM_MINMAX);
		watershed_result.convertTo(watershed_result, CV_8U);
		aia::imshow("Internal + external markers image", watershed_result, true, 2.0f);
		
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