// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	std::string win_name = "Gradient edge detection demo";
	int sigma_x10 = 10;
	int T_high = 100;

	void CannyEdgeDetection(int pos, void* userdata)
	{
		float sigma = sigma_x10 / 10.0f;
		int k = 6 * sigma;
		if (k % 2 == 0)
			k += 1;
		cv::Mat img_blurred;
		cv::GaussianBlur(img, img_blurred, cv::Size(k, k), sigma, sigma);

		cv::Mat edge;
		cv::Canny(img_blurred, edge, T_high, T_high / 3);

		cv::imshow("Original image", img);
		cv::imshow(win_name, edge);
	}
};



int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/brain_mri.jpg", cv::IMREAD_GRAYSCALE);


	cv::namedWindow(win_name);
	cv::createTrackbar("sigma", win_name, &sigma_x10, 100, CannyEdgeDetection);
	cv::createTrackbar("T high", win_name, &T_high, 300, CannyEdgeDetection);
	CannyEdgeDetection(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}