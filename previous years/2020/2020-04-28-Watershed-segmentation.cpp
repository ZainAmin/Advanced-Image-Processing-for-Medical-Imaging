// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

namespace
{
	cv::Mat img;
	cv::Mat img_gray;
	cv::Mat img_bin;
	cv::Mat external_markers;

	std::string win_name = "Watershed";
	int dist_transf_threshold_x10 = 0;

	void watershed_segmentation(int pos, void* userdata)
	{
		cv::Mat dist_tranf;
		cv::distanceTransform(img_bin, dist_tranf, CV_DIST_L1, 3);
		cv::threshold(dist_tranf, dist_tranf, dist_transf_threshold_x10/10.0f, 255, CV_THRESH_BINARY);
		dist_tranf.convertTo(dist_tranf, CV_8U);

		cv::Mat internal_markers = dist_tranf;
		std::vector < std::vector < cv::Point> > contours;
		cv::findContours(internal_markers, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		internal_markers.setTo(cv::Scalar(0));
		for(int k=0; k<contours.size(); k++)
			cv::drawContours(internal_markers, contours, k, cv::Scalar(k+2), CV_FILLED);

		cv::Mat markers = external_markers + internal_markers;
		markers.convertTo(markers, CV_32S);

		cv::watershed(img, markers);

		cv::Mat watershed_result = markers.clone();
		// -1*255 = -255 (dams)
		// (>0)*255 = 255 (regions)
		// + 255 --> 0 (dams) and 255 (regions)
		// invert --> 255 (dams) and 0 (regions)
		// extract contours --> dams
		/*cv::Mat dams;
		watershed_result.convertTo(dams, CV_8U, 255, 255);
		dams = 255 - dams;
		contours.clear();
		cv::findContours(dams, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		cv::drawContours(img, contours, -1, cv::Scalar(0, 255, 255), 2, CV_AA);*/

		markers = cv::abs(markers);
		cv::normalize(markers, markers, 0, 255, cv::NORM_MINMAX);
		markers.convertTo(markers, CV_8U);

		cv::resize(markers, markers, cv::Size(0,0), 2, 2, cv::INTER_CUBIC);

		cv::imshow(win_name, markers);
	}
}
		
// GOAL: Pectoral Muscle Segmentation with Mean-Shift
int main() 
{
	try
	{
		// load image
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pills.jpg");
		if(!img.data)
			throw aia::error("Cannot open image");

		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		cv::threshold(img_gray, img_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		cv::morphologyEx(img_bin, img_bin, 
			cv::MORPH_OPEN, 
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));


		cv::dilate(img_bin, external_markers, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));
		std::vector < std::vector < cv::Point> > contours;
		cv::findContours(external_markers, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		external_markers.setTo(cv::Scalar(0));
		cv::drawContours(external_markers, contours, -1, cv::Scalar(1));

		aia::imshow("Image binarized", img_bin, true, 2);
		aia::imshow("External markers", external_markers, true, 2);

		// launch GUI application
		cv::namedWindow(win_name);
		cv::createTrackbar("distance threshold", win_name, &dist_transf_threshold_x10, 100, watershed_segmentation);
		watershed_segmentation(0, 0);
		cv::waitKey(0);

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
