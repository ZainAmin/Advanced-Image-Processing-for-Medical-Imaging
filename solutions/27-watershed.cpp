// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main() 
{
	try
	{
		// load image
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pills.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");

		float scaling_factor = 2.0;
		aia::imshow("Pills", img, true, scaling_factor);

		// binarization
		cv::Mat binarized;
		cv::Mat img_gray;
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		cv::threshold(img_gray, binarized, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		aia::imshow("Pills binarized", binarized, true, scaling_factor);

		// cleaning
		cv::Mat opened;
		cv::morphologyEx(binarized, opened, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));
		aia::imshow("Pills opened", opened, true, scaling_factor);

		// distance transform
		cv::Mat distance_transform;
		cv::distanceTransform(opened, distance_transform, CV_DIST_L1, CV_DIST_MASK_PRECISE);
		cv::normalize(distance_transform, distance_transform, 0, 255, cv::NORM_MINMAX);
		distance_transform.convertTo(distance_transform, CV_8U);
		aia::imshow("Distance Transform", distance_transform, true, scaling_factor);

		// thresholding distance transform
		// = we have internal markers
		cv::Mat internal_markers;
		cv::threshold(distance_transform, internal_markers, 180, 255, CV_THRESH_BINARY);
		aia::imshow("Distance Transform", internal_markers, true, scaling_factor);

		// display internal markers
		cv::Mat img_copy = img.clone();
		img_copy.setTo(cv::Scalar(0,0,255), internal_markers);
		aia::imshow("Pills with internal markers", img_copy, true, scaling_factor);

		// display external markers
		cv::Mat binarized_dilated;
		cv::dilate(binarized, binarized_dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15)));
		cv::Mat external_marker = 255-binarized_dilated;
		aia::imshow("External marker", binarized_dilated, true, scaling_factor);

		// build the marker image to be inputted to the whatershed
		cv::Mat markers(img.rows, img.cols, CV_32S, cv::Scalar(0));
		std::vector < std::vector <cv::Point> > contours;
		cv::findContours(internal_markers, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		for(int i=0; i<contours.size(); i++)
			cv::drawContours(markers, contours, i, cv::Scalar(i+1), CV_FILLED);
		markers.setTo(cv::Scalar(contours.size()+1), external_marker);

		// can we visualize the markers?
		cv::Mat markers_image = markers.clone();
		cv::normalize(markers_image, markers_image, 0, 255, cv::NORM_MINMAX);
		markers_image.convertTo(markers_image, CV_8U);
		aia::imshow("Markers image", markers_image, true, scaling_factor);

		cv::watershed(img, markers);

		// marker = -1 where there are dams
		// marker + 1 = 0 where there are dams, != 0 in the rest of the image
		markers += 1;
		markers.convertTo(markers, CV_8U);
		cv::threshold(markers, markers, 0, 255, CV_THRESH_BINARY_INV);
		aia::imshow("Dams = region contours", markers, true, scaling_factor);

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
