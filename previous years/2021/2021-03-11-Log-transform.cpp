// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main() 
{
	try
	{
		// load the image
		cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram.tif", CV_LOAD_IMAGE_UNCHANGED);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		int true_depth = ucas::imdepth_detect(input_img);
		printf("True depth = %d\n", true_depth);
		aia::imshow("Input mammogram", input_img*4, true, 0.3);

		int L = std::pow(2, true_depth) - 1;
		float c = (L-1)/log(L);
		for(int y=0; y<input_img.rows; y++)
		{
			unsigned short* yRow = input_img.ptr<unsigned short>(y);
			for(int x=0; x<input_img.cols; x++)
				yRow[x] = ucas::round<float>(c*log(1 + yRow[x]));
		}

		aia::imshow("Enhanced mammogram", 65535 - input_img*4, true, 0.3);
		cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/raw_mammogram_log.tif", 65535 - input_img*4);

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
