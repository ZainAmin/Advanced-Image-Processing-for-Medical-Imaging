// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

namespace
{
	cv::Mat input_img;
	std::string win_name;
	int k_x10 = 0;

	void sharpenLaplacianCallback(int pos, void* userdata)
	{
		float k = k_x10/10.0f;
		cv::Mat sharpKernel = (cv::Mat_<float>(3, 3) <<
			-k, -k,    -k,
			-k, 1+8*k, -k,
			-k, -k,    -k);

		cv::Mat sharpened_img;
		cv::filter2D(input_img, sharpened_img, CV_32F, sharpKernel);

		// WARNING: normalizing image is not required (and should not be done) in this case
		// since we want to cut the top and bottom spikes caused by edge stretching...
		//cv::normalize(sharpened_img, sharpened_img, 0, 255, cv::NORM_MINMAX);
		sharpened_img.convertTo(sharpened_img, CV_8U);

		cv::imshow(win_name, input_img);
		cv::imshow("Sharpened image", sharpened_img);
	}
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/eye.blurry.png");
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		// create the UI
		win_name = "Sharpen";
		cv::namedWindow(win_name);
		cv::createTrackbar("k", win_name, &k_x10, 100, sharpenLaplacianCallback);

		// start the UI
		sharpenLaplacianCallback(0, 0);

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
