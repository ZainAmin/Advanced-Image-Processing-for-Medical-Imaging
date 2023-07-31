// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// <>

cv::Mat medianCustom(const cv::Mat& img, int k)
{
	int h = k / 2;
	int m = (k * k) / 2;

	std::vector<unsigned char> buffer(k * k);
	cv::Mat out_img(img.rows, img.cols, CV_8U, cv::Scalar(0));

	for (int i = h; i < img.rows - h; i++)
	{
		unsigned char* out_iRow = out_img.ptr(i);
		for (int j = h; j < img.cols - h; j++)
		{
			int idx = 0;
			for (int ii = i - h; ii < i + h; ii++)
			{
				const unsigned char* buffer_iRow = img.ptr(ii);
				for (int jj = j - h; jj < j + h; jj++)
					buffer[idx++] = buffer_iRow[jj];
			}
			std::nth_element(buffer.begin(), buffer.begin() + m, buffer.end());
			out_iRow[j] = buffer[m];
		}
	}

	return out_img;
}

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/rice.png", cv::IMREAD_GRAYSCALE);
	aia::imshow("Original image", img, true, 2.0);

	ucas::Timer timer;
	cv::Mat res1 = medianCustom(img, 5);
	printf("Median (custom): elapsed time = %.3f s\n", timer.elapsed<float>());
	aia::imshow("Median-filtered (custom)", res1, true, 2.0);

	timer.restart();
	cv::Mat res2;
	cv::medianBlur(img, res2, 5);
	printf("Median (OpenCV): elapsed time = %.3f s\n", timer.elapsed<float>());
	aia::imshow("Median-filtered (OpenCV)", res2, true, 2.0);

	return EXIT_SUCCESS;
}