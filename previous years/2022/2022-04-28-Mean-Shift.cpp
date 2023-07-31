// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	// load image
	cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina.png");

	cv::Mat result;
	cv::pyrMeanShiftFiltering(input_img, result, 2, 30, 0);

	aia::imshow("Original", input_img);
	aia::imshow("MS result", result);

	cv::imwrite(std::string(EXAMPLE_IMAGES_PATH) + "/retina.MS.png", result);

	return EXIT_SUCCESS;
}
