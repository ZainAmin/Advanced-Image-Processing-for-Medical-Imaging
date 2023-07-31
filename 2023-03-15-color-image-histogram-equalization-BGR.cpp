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
	std::vector <cv::Mat> img_chans(3);
	cv::split(img, img_chans);

	std::vector<int> img_histB = ucas::histogram(img_chans[0]); // luminance channel
	std::vector<int> img_histG = ucas::histogram(img_chans[1]); // luminance channel
	std::vector<int> img_histR = ucas::histogram(img_chans[2]); // luminance channel
	aia::imshow("Original image", img);
	
	std::vector<unsigned char> hist_eq_LUT_B(256);
	std::vector<unsigned char> hist_eq_LUT_G(256);
	std::vector<unsigned char> hist_eq_LUT_R(256);
	int acc = 0;
	float scale_f = 255.0f / (img.rows * img.cols);
	for (int k = 0; k < img_histB.size(); k++)
	{
		acc += img_histB[k];
		hist_eq_LUT_B[k] = ucas::round(scale_f * acc);
	}
	acc = 0;
	for (int k = 0; k < img_histG.size(); k++)
	{
		acc += img_histG[k];
		hist_eq_LUT_G[k] = ucas::round(scale_f * acc);
	}
	acc = 0;
	for (int k = 0; k < img_histR.size(); k++)
	{
		acc += img_histR[k];
		hist_eq_LUT_R[k] = ucas::round(scale_f * acc);
	}

	for (int i = 0; i < img.rows; i++)
	{
		unsigned char* ithRowB = img_chans[0].ptr(i);
		unsigned char* ithRowG = img_chans[1].ptr(i);
		unsigned char* ithRowR = img_chans[2].ptr(i);
		for (int j = 0; j < img.cols; j++)
		{
			ithRowB[j] = hist_eq_LUT_B[ithRowB[j]];
			ithRowG[j] = hist_eq_LUT_G[ithRowG[j]];
			ithRowR[j] = hist_eq_LUT_R[ithRowR[j]];
		}
	}

	cv::Mat out_img;
	cv::merge(img_chans, out_img);
	aia::imshow("Contrast enhanced image (HQ)", out_img);

	return EXIT_SUCCESS;
}