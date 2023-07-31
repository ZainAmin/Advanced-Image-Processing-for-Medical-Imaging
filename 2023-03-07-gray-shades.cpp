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
	cv::Mat img(5120, 5120, CV_8U, cv::Scalar(0));

	ucas::Timer timer;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img.at<unsigned char>(i, j) = (i/float(img.rows-1))*255;
	printf("Approach at(): elapsed time = %.3f s\n", timer.elapsed<float>());

	timer.restart();
	for (int i = 0; i < img.rows; i++)
	{
		unsigned char* iThRow = img.ptr(i);
		for (int j = 0; j < img.cols; j++)
			iThRow[j] = (i / float(img.rows - 1)) * 255;
	}
	printf("Approach row-access(): elapsed time = %.3f s\n", timer.elapsed<float>());

	aia::imshow("Display image", img);

	return EXIT_SUCCESS;
}