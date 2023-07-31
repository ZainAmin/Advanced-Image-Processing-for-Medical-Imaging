// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

using namespace ucas;
using namespace aia;

int main()
{
	// load image
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/pro_mammogram_small.tif");
	
	cv::Mat result;
	cv::pyrMeanShiftFiltering(img, result, 20, 20, 0);

	aia::imshow("Original image", img, true, 2.0f);
	aia::imshow("Mean-shift result", result, true, 2.0f);
	
	return EXIT_SUCCESS;
}
