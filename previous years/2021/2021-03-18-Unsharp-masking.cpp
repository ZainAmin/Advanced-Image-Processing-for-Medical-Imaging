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

	void unsharpMaskingCallback(int pos, void* userdata)
	{
		float k = k_x10/10.0f;
		
		cv::Mat low_freq_img;
		cv::GaussianBlur(input_img, low_freq_img, cv::Size(7,7), 0, 0);

		cv::Mat sharpened_img = input_img.clone();
		sharpened_img = input_img + k*(input_img-low_freq_img);

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
		cv::createTrackbar("k", win_name, &k_x10, 100, unsharpMaskingCallback);

		// start the UI
		unsharpMaskingCallback(0, 0);

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
