// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

int main()
{
	cv::Mat img(512, 512, CV_8U, cv::Scalar(0));
	aia::imshow("Image", img);

	ucas::Timer timer;

	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
			img.at<unsigned char>(y, x) = float(y) / (img.rows - 1) * 255;
	}
	printf("Elapsed time (slow version) = %.3f\n", timer.elapsed<float>());

	timer.restart();
	for (int y = 0; y < img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);
		for (int x = 0; x < img.cols; x++)
			yRow[x] = float(y) / (img.rows - 1) * 255;
	}

	printf("Elapsed time (fast version) = %.3f\n", timer.elapsed<float>());
	aia::imshow("Image", img);

	return EXIT_SUCCESS;
	
}

