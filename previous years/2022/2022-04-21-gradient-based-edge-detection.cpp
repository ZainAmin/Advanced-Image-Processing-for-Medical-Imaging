// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
	cv::Mat img;
	std::string win_name = "Gradient edge detection demo";
	int grad_mag_thresh = 50;

	void gradientEdgeDetection(int pos, void* userdata)
	{


		cv::Mat dx, dy;
		cv::Sobel(img, dx, CV_32F, 1, 0);
		cv::Sobel(img, dy, CV_32F, 0, 1);
		cv::Mat mag;
		cv::magnitude(dx, dy, mag);

		double minV, maxV;
		cv::minMaxLoc(mag, &minV, &maxV);
		cv::threshold(mag, mag, (grad_mag_thresh / 100.0) * maxV, 255, cv::THRESH_BINARY);

		mag.convertTo(mag, CV_8U);
		cv::imshow(win_name, mag);
	}
};



int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina.png", cv::IMREAD_GRAYSCALE);
	cv::GaussianBlur(img, img, cv::Size(7, 7), 0, 0);

	cv::namedWindow(win_name);
	cv::createTrackbar("grad mag", win_name, &grad_mag_thresh, 100, gradientEdgeDetection);
	gradientEdgeDetection(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}