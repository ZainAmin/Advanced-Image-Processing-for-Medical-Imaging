// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"


int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", cv::IMREAD_GRAYSCALE);
	cv::Mat bin;
	cv::threshold(img, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	aia::imshow("Binarized on original", bin, true, 2.0);

	cv::Mat IF;
	cv::morphologyEx(img, IF, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40)));
	aia::imshow("IF", IF, true, 2.0);

	cv::Mat th = img - IF;
	aia::imshow("Tophat", th, true, 2.0);
	//cv::threshold(th, th, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	//aia::imshow("Binarized after tophat", th, true, 2.0);

	cv::Scalar avgIF = cv::mean(IF);
	aia::imshow("Original image", img, true, 2.0);
	aia::imshow("Corrected image", th + avgIF, true, 2.0);


	return EXIT_SUCCESS;
}