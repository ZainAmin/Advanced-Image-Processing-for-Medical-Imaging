// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

// GOAL: load an image and reduce the gray levels down to
//       a number of levels specified by the user
int main()
{
	// load the image
	cv::Mat input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/tools.png", cv::IMREAD_GRAYSCALE);

	aia::imshow("Original image", input_img);
	aia::imshow("Histogram", ucas::imhist(input_img));

	// OTSU: not so good choice since histogram is not bimodal
	int T = ucas::getOtsuAutoThreshold(ucas::histogram(input_img));
	printf("Otsu T = %d\n", T);
	cv::Mat binarized_img;
	cv::threshold(input_img, binarized_img, T, 255, cv::THRESH_BINARY);
	aia::imshow("Otsu-binarized image", binarized_img);

	// TRIANGLE: better choice since dark background is dominant
	T = ucas::getTriangleAutoThreshold(ucas::histogram(input_img));
	printf("Triangle T = %d\n", T);
	cv::threshold(input_img, binarized_img, T, 255, cv::THRESH_BINARY);
	aia::imshow("Triangle-binarized image", binarized_img);

	// waits for user to press a button and exit from the app
	cv::waitKey(0);

	return EXIT_SUCCESS;
}
