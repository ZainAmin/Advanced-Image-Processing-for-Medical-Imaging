// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	std::string win_name = "Sharpening";
	int k_x10 = 0;
	int sigma_x10 = 10;

	void unsharpMasking(int pos, void* userdata)
	{
		float k = k_x10 / 10.0f;
		float sigma = sigma_x10 / 10.0f;

		cv::Mat img_blurred;
		cv::GaussianBlur(img, img_blurred, cv::Size(-1, -1), sigma, sigma);
		cv::Mat img_sharpened = img + k * (img - img_blurred);
	
		cv::imshow(win_name, img_sharpened);
	}
};



int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/eye.blurry.png");
	cv::resize(img, img, cv::Size(-1, -1), 2, 2);

	cv::namedWindow(win_name);
	cv::createTrackbar("sigma", win_name, &sigma_x10, 100, unsharpMasking);
	cv::createTrackbar("k", win_name, &k_x10, 100, unsharpMasking);
	unsharpMasking(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}