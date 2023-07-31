// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main()
{
	cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/road.jpg");

	cv::Mat res(img.rows, img.cols, CV_32FC3, cv::Scalar(0, 0, 0));
	int angles = 16;
	for (int ang_idx = 0; ang_idx < angles; ang_idx++)
	{
		float theta = ang_idx * ucas::PI / angles;

		cv::Mat kernel_re = cv::getGaborKernel(cv::Size(31, 31), 2, theta, 5, 0.5, 0);
		cv::Mat kernel_im = cv::getGaborKernel(cv::Size(31, 31), 2, theta, 5, 0.5);

		cv::Mat filtered_re, filtered_im;
		cv::filter2D(img, filtered_re, CV_32FC3, kernel_re);
		cv::filter2D(img, filtered_im, CV_32FC3, kernel_im);
		cv::Mat filtered_mag;
		cv::magnitude(filtered_re, filtered_im, filtered_mag);

		res = cv::max(res, filtered_mag);

		cv::normalize(kernel_re, kernel_re, 0, 1, cv::NORM_MINMAX);
		cv::normalize(kernel_im, kernel_im, 0, 1, cv::NORM_MINMAX);
		cv::normalize(filtered_mag, filtered_mag, 0, 1, cv::NORM_MINMAX);
		aia::imshow("kernel re", kernel_re, false, 6.0);
		aia::imshow("kernel img", kernel_im, false, 6.0);
		aia::imshow("filtered mag", filtered_mag, true);
	}

	cv::normalize(res, res, 0, 255, cv::NORM_MINMAX);
	res.convertTo(res, CV_8UC3);
	aia::imshow("Result", res, true, 1.0);

	return EXIT_SUCCESS;
}
