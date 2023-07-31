// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	// since we work with a GUI, one possible solution to pass parameters
	// to/from the GUI/application core functions is to store parameters 
	// (including images) in global variables
	cv::Mat img;
	cv::Mat img_gray;
	cv::Mat img_bin;
	std::vector <std::vector <cv::Point> > contours;

	void aMouseCallback(int event, int x, int y, int, void* userdata)
	{
		if(event == cv::EVENT_LBUTTONDBLCLK)
		{
			int object_idx = -1;
			for(int k=0; k<contours.size(); k++)
				if(cv::pointPolygonTest(contours[k], cv::Point2f(x, y), false ) >= 0)
				{
					object_idx = k;
					break;
				}

			if(object_idx != -1)
			{
				printf("Object %d of %d\n", object_idx+1, contours.size());
				cv::Mat selection_layer = img.clone();
				cv::drawContours(selection_layer, contours, object_idx, cv::Scalar(255, 0, 0), CV_FILLED);
				cv::addWeighted(img, 0.7, selection_layer, 0.3, 0, selection_layer);
			
				cv::imshow("MagicWand", selection_layer);
			}
		}

	}
}


int main() 
{
	try
	{	
		// load the image in grayscale
		std::string img_name = "tools.bmp";
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);
		if(!img.data)
			throw ucas::Error("cannot load image");

		// convert to gray
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

		// binarize
		std::vector<int> hist = ucas::histogram(img_gray);
		//int T = ucas::getOtsuAutoThreshold(hist);
		int T = ucas::getTriangleAutoThreshold(hist);
		cv::threshold(img_gray, img_bin, T, 255, CV_THRESH_BINARY);

		/*cv::adaptiveThreshold(img_gray, img_bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 51, 10);
		ucas::imshow("Binarized", img_bin, true, 2.0);*/

		// extract connected components
		std::vector <std::vector <cv::Point> > all_contours;
		cv::findContours(img_bin, all_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		/*img_bin.setTo(cv::Scalar(0));
		cv::drawContours(img_bin, contours, -1, cv::Scalar(255), 2, CV_AA);
		ucas::imshow("contours", img_bin);*/

		// optionally filter small components
		for(int k=0; k<all_contours.size(); k++)
			if(cv::contourArea(all_contours[k]) > 100)
				contours.push_back(all_contours[k]);

		// create the window and insert the trackbar(s)
		cv::namedWindow("MagicWand");
		cv::imshow("MagicWand", img);
		cv::setMouseCallback("MagicWand", aMouseCallback);

		// wait for key press = windows stay opened until the user presses any key
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