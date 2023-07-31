// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lowcontrast.png", cv::IMREAD_GRAYSCALE);
	aia::imshow("Image original", img);
	aia::imshow("Histogram (original)", ucas::imhist(img));

	double maxv, minv;
	cv::minMaxLoc(img, &minv, &maxv);
	img = ((img - minv) / (maxv - minv)) * 255;

	// alternatively, use OpenCV normalization
	//cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);

	aia::imshow("Image normalized", img);
	aia::imshow("Histogram (after normalization)", ucas::imhist(img));

	return EXIT_SUCCESS;
}