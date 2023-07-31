// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int block_size = 91;
int C = -25;
std::string winname = "Adaptive thresholding";
cv::Mat img;
std::vector<int> hist;

void adaptiveThresholding(int pos, void* userdata)
{
	cv::imshow(winname, img);
	cv::Mat img_thresholded;

	int T_mean = ucas::getMeanThreshold(hist);
	cv::threshold(img, img_thresholded, T_mean, 255, cv::THRESH_BINARY);
	cv::imshow("Mean thresholding", img_thresholded);

	int T_Otsu = ucas::getOtsuAutoThreshold(hist);
	cv::threshold(img, img_thresholded, T_Otsu, 255, cv::THRESH_BINARY);
	cv::imshow("Otsu thresholding", img_thresholded);

	int T_triangle = ucas::getTriangleAutoThreshold(hist);
	cv::threshold(img, img_thresholded, T_triangle, 255, cv::THRESH_BINARY);
	cv::imshow("Triangle thresholding", img_thresholded);

	if (block_size % 2 == 1)
	{
		cv::adaptiveThreshold(img, img_thresholded, 255,
			cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block_size, C);
		cv::imshow(winname, img_thresholded);
	}
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png",
		cv::IMREAD_GRAYSCALE);
	cv::resize(img, img, cv::Size(0, 0), 2, 2);

	hist = ucas::histogram(img);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	cv::createTrackbar("block size", winname, &block_size, 100, adaptiveThresholding);
	cv::createTrackbar("C", winname, &C, 100, adaptiveThresholding);
	cv::setTrackbarMin("C", winname, -100);

	adaptiveThresholding(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}