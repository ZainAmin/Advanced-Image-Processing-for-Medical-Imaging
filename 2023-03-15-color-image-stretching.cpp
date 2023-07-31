// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

int perc1 = 5;
int perc2 = 95;
std::string winname = "Stretching tool";
cv::Mat img;
cv::Mat img_Lab;
std::vector<cv::Mat> img_Lab_chans(3);
std::vector<int> img_hist;

void piecewiseLinearStretching(int pos, void* userdata)
{
	cv::Mat out_L(img.rows, img.cols, CV_8U);
	/*printf("perc1 = %d\n", perc1);
	printf("perc2 = %d\n", perc2);*/

	// image histogram already calculated in the main

	// calculate percentiles
	int acc = 0;
	float perc1f = perc1 / 100.0f;
	int nPixels = img.rows * img.cols;
	int perc1_value = 0;
	for (int k = 0; k < img_hist.size(); k++)
	{
		acc += img_hist[k];
		if (acc >= perc1f * nPixels)
		{
			perc1_value = k;
			break;
		}
	}
	float perc2f = perc2 / 100.0f;
	int perc2_value = 255;
	acc = 0;
	for (int k = 0; k < img_hist.size(); k++)
	{
		acc += img_hist[k];
		if (acc >= perc2f * nPixels)
		{
			perc2_value = k;
			break;
		}
	}

	int c = 0.05 * 255;
	int d = 0.95 * 255;
	for (int i = 0; i < img.rows; i++)
	{
		unsigned char* iThRowIn = img_Lab_chans[0].ptr(i);
		unsigned char* iThRowOut = out_L.ptr(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (iThRowIn[j] < perc1_value)
				iThRowOut[j] = iThRowIn[j] * c / perc1_value;
			else if (iThRowIn[j] > perc2_value)
				iThRowOut[j] = (iThRowIn[j] - perc2_value) * (255 - d) / (255 - perc2_value) + d;
			else
				iThRowOut[j] = (iThRowIn[j] - perc1_value) * (d - c) / (perc2_value - perc1_value) + c;
		}
	}

	std::vector<cv::Mat> out_Lab(3);
	out_Lab[0] = out_L;
	out_Lab[1] = img_Lab_chans[1];
	out_Lab[2] = img_Lab_chans[2];
	cv::Mat out_img;
	cv::merge(out_Lab, out_img);
	cv::cvtColor(out_img, out_img, cv::COLOR_Lab2BGR);
	cv::imshow(winname, out_img);
}

int main()
{
	img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/low-contrast-elk.jpg");
	cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
	cv::split(img_Lab, img_Lab_chans);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Perc 1", winname, &perc1, 100, piecewiseLinearStretching);
	cv::createTrackbar("Perc 2", winname, &perc2, 100, piecewiseLinearStretching);

	img_hist = ucas::histogram(img_Lab_chans[0]); // luminance channel

	piecewiseLinearStretching(0, 0);
	cv::waitKey(0);

	return EXIT_SUCCESS;
}