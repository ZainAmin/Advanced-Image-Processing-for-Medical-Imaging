// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
		
namespace
{
	cv::Mat input_img;
	std::string win_name = "Watershed";
	int threshold = 75;

	void watershedCallback(int pos, void* userdata)
	{
		cv::Mat img_copy = input_img.clone();

		// binarization
		cv::Mat img_gray;
		cv::cvtColor(img_copy, img_gray, cv::COLOR_BGR2GRAY);
		cv::Mat img_bin;
		cv::threshold(img_gray, img_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		//aia::imshow("Binarized", img_bin);

		// distance transform
		cv::Mat img_dist;
		cv::distanceTransform(img_bin, img_dist, CV_DIST_L2, 3);
		cv::normalize(img_dist, img_dist, 0, 255, cv::NORM_MINMAX);
		img_dist.convertTo(img_dist, CV_8U);
		//aia::imshow("Distance transform image", img_dist);

		// internal markers (seeds) selection
		cv::Mat img_seeds;
		cv::threshold(img_dist, img_seeds, 255*(threshold/100.0f), 255, cv::THRESH_BINARY);
		//aia::imshow("Seeds", img_seeds);

		// remove very small seeds that perhaps occur along bridges between neighboring pills
		cv::morphologyEx(img_seeds, img_seeds, cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
		//aia::imshow("Seeds after opening", img_seeds);

		// prepare markers image
		std::vector < std::vector <cv::Point> > objects;
		cv::findContours(img_seeds, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		cv::Mat markers(input_img.rows, input_img.cols, CV_32S, cv::Scalar(0));
		for(int k=0; k<objects.size(); k++)
			cv::drawContours(markers, objects, k, cv::Scalar(k+1), CV_FILLED);
		int internal_marker_n = objects.size();

		// add the external marker
		cv::Mat external_marker;
		cv::dilate(img_bin, external_marker, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(51, 51)));
		objects.clear();
		cv::findContours(external_marker, objects, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		int largest_area_idx = -1;
		double largest_area = 0;
		for(int k=0; k<objects.size(); k++)
		{
			double A = cv::contourArea(objects[k]);
			if(A > largest_area)
			{
				largest_area = A;
				largest_area_idx = k;
			}
		}
		cv::drawContours(img_bin, objects, largest_area_idx, 128, 2);

		//cv::drawContours(img_bin, objects, largest_area_idx, 128, 2);
		//aia::imshow("Debugging external", img_bin);


		// for debugging purposes: generate a visualizable version of the marker image
		cv::Mat markers_vis;
		cv::normalize(markers, markers_vis, 0, 255, cv::NORM_MINMAX);
		markers_vis.convertTo(markers_vis, CV_8U);
		//aia::imshow("Markers", markers_vis);

		// apply the marker-controlled watershed
		cv::watershed(img_copy, markers);

		//  -1 = dams
		//   0 = there are not zeros
		// !=0 = regions

		// if we multiply by 255

		//   -255 = dams
		//      
		// >= 255 = regions

		// if we add 255

		//       0 = dams
		//   > 255 = regions

		// if we convert to 8 bits unsigned grayscale

		//       0 = dams
		//     255 = regions
		markers = markers*255 + 255;
		markers.convertTo(markers, CV_8U);

		//    255  = dams (region contours) = segmentation
		//      0  = regions 
		markers = 255 - markers;

		cv::dilate(markers, markers, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
		img_copy.setTo(cv::Scalar(0, 0, 255), markers);

		cv::imshow(win_name, img_copy);
	}
}

int main() 
{
	try
	{
		// load image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pills3.jpeg");
		if(!input_img.data)
			throw aia::error("Cannot open image");

		// create the UI
		cv::namedWindow(win_name);
		cv::createTrackbar("thresh", win_name, &threshold, 100,  watershedCallback);
		
		// start the UI
		watershedCallback(0, 0);

		// waits for user to press a button and exit from the app
		cv::waitKey(0);
		

		return EXIT_SUCCESS;
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
