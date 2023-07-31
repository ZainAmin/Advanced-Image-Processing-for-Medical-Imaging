// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/retina_lowcontrast.bmp");
	cv::Mat img_Lab;
	//cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
	cv::cvtColor(img, img_Lab, cv::COLOR_BGR2HSV);
	std::vector <cv::Mat> img_Lab_chans(3);
	cv::split(img_Lab, img_Lab_chans);

	std::vector<int> img_hist = ucas::histogram(img_Lab_chans[2]); // luminance channel
	aia::imshow("Original image", img);
	aia::imshow("Original histogram", ucas::imhist(img_Lab_chans[0]));

	std::vector<unsigned char> hist_eq_LUT(256);
	int acc = 0;
	float scale_f = 255.0f / (img.rows * img.cols);
	for (int k = 0; k < img_hist.size(); k++)
	{
		acc += img_hist[k];
		hist_eq_LUT[k] = ucas::round(scale_f * acc);
	}

	for (int i = 0; i < img.rows; i++)
	{
		unsigned char* ithRow = img_Lab_chans[2].ptr(i);
		for (int j = 0; j < img.cols; j++)
			ithRow[j] = hist_eq_LUT[ithRow[j]];
	}

	cv::Mat out_img;
	cv::merge(img_Lab_chans, out_img);
	//cv::cvtColor(out_img, out_img, cv::COLOR_Lab2BGR);
	cv::cvtColor(out_img, out_img, cv::COLOR_HSV2BGR);
	aia::imshow("Contrast enhanced image (HQ)", out_img);
	aia::imshow("Equalized histogram", ucas::imhist(img_Lab_chans[2]));

	return EXIT_SUCCESS;
}