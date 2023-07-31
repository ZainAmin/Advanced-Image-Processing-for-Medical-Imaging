// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int k = 0;
std::string winname = "Sharpening with Laplacian";
cv::Mat img;

void unsharpMasking(int pos, void* userdata)
{
	cv::Mat img_blurred;
	cv::GaussianBlur(img, img_blurred, cv::Size(0, 0), 1, 1);

	cv::Mat out_img = img + k * (img - img_blurred);

	cv::imshow(winname, out_img);
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/eye.blurry.png");
	cv::resize(img, img, cv::Size(0, 0), 2, 2);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("k", winname, &k, 20, unsharpMasking);

	unsharpMasking(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}