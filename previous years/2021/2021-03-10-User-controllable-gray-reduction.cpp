// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"


// advanced gray level reduction
// corrects pure black (0) and pure white (255) mapping
cv::Mat reduceGrayLevelsAdvanced(cv::Mat & img, int L)
{
	int step = 256/L;
	int max_quantized_value = 255/step;
	for(int y=0; y<img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for(int x=0; x<img.cols; x++)
			yRow[x] = ((yRow[x] / step) * 255) / max_quantized_value;
	}

	return img;
}

namespace
{
	cv::Mat input_img;
	std::string win_name;
	int L;

	void greyLevelReductionCallback(int pos, void* userdata)
	{
		if(L > 1)
		{
			cv::Mat img_copy = input_img.clone();
			cv::imshow(win_name, reduceGrayLevelsAdvanced(img_copy, L));
		}
	}
}

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		// create the UI
		win_name = "Reduce gray levels";
		cv::namedWindow(win_name);
		cv::createTrackbar("Gray levels", win_name, &L, 100, greyLevelReductionCallback);

		// start the UI
		greyLevelReductionCallback(0, 0);

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
