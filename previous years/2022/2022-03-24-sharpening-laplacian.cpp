// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	std::string win_name = "Sharpening";
	int k_x10 = 0;

	void sharpenLaplacian(int pos, void* userdata)
	{
		float k = k_x10 / 10.0f;

		cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
			-k,		-k,		-k,
			-k,		1+8*k,	-k,
			-k,		-k,		-k);

		cv::Mat img_sharpened;


		cv::filter2D(img, img_sharpened, CV_8U, kernel);

		// this will cause the image to become more and more gray
		//cv::filter2D(img, img_sharpened, CV_32F, kernel);
		//cv::normalize(img_sharpened, img_sharpened, 0, 255, cv::NORM_MINMAX);
		//img_sharpened.convertTo(img_sharpened, CV_8U);
	
		cv::imshow(win_name, img_sharpened);
	}
};



int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/eye.blurry.png");
	cv::resize(img, img, cv::Size(-1, -1), 2, 2);

	cv::namedWindow(win_name);
	cv::createTrackbar("k", win_name, &k_x10, 100, sharpenLaplacian);
	sharpenLaplacian(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}