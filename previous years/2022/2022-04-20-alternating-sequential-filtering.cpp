// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"


int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img_copy = img.clone();

	for (int i = 3; ; i += 2)
	{
		cv::Mat SE = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i));
		cv::morphologyEx(img, img, cv::MORPH_OPEN, SE);
		cv::morphologyEx(img, img, cv::MORPH_CLOSE, SE);

		aia::imshow("Original", img_copy, true, 2);
		aia::imshow("ASF denoised", img, true, 2);
		aia::imshow("Difference", cv::abs(img_copy-img), true, 2);
	}

	return EXIT_SUCCESS;
}