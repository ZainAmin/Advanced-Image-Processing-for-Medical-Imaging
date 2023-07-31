// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int gamma_x10 = 10;
std::string winname = "Gamma correction";
cv::Mat img;

void gammaCorrection(int pos, void* userdata)
{
	cv::Mat out_img(img.rows, img.cols, CV_8U);
	
	float gamma = gamma_x10 / 10.0f;
	float c = std::pow(255, 1 - gamma);
	for (int i = 0; i < img.rows; i++)
	{
		unsigned char* iThRowIn = img.ptr(i);
		unsigned char* iThRowOut = out_img.ptr(i);
		for (int j = 0; j < img.cols; j++)
			iThRowOut[j] = c * std::pow(iThRowIn[j], gamma);
	}

	cv::imshow(winname, out_img);
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lightning_gray.jpg",
		cv::IMREAD_GRAYSCALE);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Gamma", winname, &gamma_x10, 100, gammaCorrection);
	
	gammaCorrection(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}