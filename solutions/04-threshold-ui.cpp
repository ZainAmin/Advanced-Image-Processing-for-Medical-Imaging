// include aia and ucas utilities
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	// since we work with a GUI, one possible solution is to store parameters 
	// (including images) in global variables
	cv::Mat img;									// original image
	int threshold;									// threshold


	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void thresholdCallback(int pos, void* userdata) 
	{

		cv::Mat thresholded;

		// convert to grayscale - thresholding is meaningless for color images
		cv::cvtColor(img, thresholded, cv::COLOR_BGR2GRAY);

		// threshold
		cv::threshold(thresholded, thresholded, threshold, 255, CV_THRESH_BINARY);

		// show result
		cv::imshow("lena", thresholded);
	}

}


int main() 
{

	try
	{
		// load the image
		img = cv::imread(EXAMPLE_IMAGES_PATH + std::string("/lena.png"), CV_LOAD_IMAGE_UNCHANGED);
		if(!img.data)
			throw aia::error("image not loaded");

		// set default parameters
		threshold = 0;

		// create the window and insert the trackbar
		cv::namedWindow("lena");
		cv::createTrackbar("threshold", "lena", &threshold, 256, thresholdCallback);

		// run thresholding the first time with default parameters
		thresholdCallback(0,0);

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

