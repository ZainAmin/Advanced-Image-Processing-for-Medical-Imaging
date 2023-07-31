// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img = cv::imread(
		std::string(EXAMPLE_IMAGES_PATH) + "/retina.png");

	aia::imshow("Original image", img);

	int k = 3;
	while (1)
	{
		cv::morphologyEx(img, img, cv::MORPH_OPEN,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));

		cv::morphologyEx(img, img, cv::MORPH_CLOSE,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));

		aia::imshow("Alternating sequential filtering", img);

		cv::imwrite(ucas::strprintf("%s/retina.regularized_%d.png", EXAMPLE_IMAGES_PATH, k), img);

		k += 2;
	}

	return EXIT_SUCCESS;
}