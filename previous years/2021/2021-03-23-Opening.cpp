// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	cv::Mat input_img;
	cv::Mat binarized_img;
	std::string win_name = "Opening";

	int k = 3;

	void openingCallback(int pos, void* userdata)
	{
		cv::Mat filtered_img;
		cv::morphologyEx(binarized_img, filtered_img, cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));

		cv::imshow("Original image", binarized_img);
		cv::imshow(win_name, filtered_img);
	}
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.bmp");
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		cv::Mat input_img_gray;
		cv::cvtColor(input_img, input_img_gray, cv::COLOR_BGR2GRAY);

		int T = ucas::getTriangleAutoThreshold(ucas::histogram(input_img_gray));
		cv::threshold(input_img_gray, binarized_img, T, 255, CV_THRESH_BINARY);

		// object selection GUI
		cv::namedWindow(win_name);
		cv::createTrackbar("k", win_name, &k, 100, openingCallback);

		// launch GUI
		openingCallback(0, 0);

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
