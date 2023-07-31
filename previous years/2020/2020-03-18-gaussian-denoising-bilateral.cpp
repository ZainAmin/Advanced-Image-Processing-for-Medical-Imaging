// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace
{
	cv::Mat img;
	cv::Mat img_float;
	cv::Mat img_gaussian_noise;
	cv::Mat img_noised;
	cv::Mat img_denoised;
	int gaussian_noise_stdev_x10 = 10;
	int median_filter_size = 3;
	int bilateral_sigma_r_x10 = 10;
	int bilateral_sigma_s_x10 = 10;
	std::string win_name;

	void gaussianNoiseReduction(int pos, void* userdata)
	{
		// add gaussian noise
		cv::randn(img_gaussian_noise, cv::Scalar(0), cv::Scalar(gaussian_noise_stdev_x10/10.0));
		img_noised = img_float + img_gaussian_noise;
		img_noised.convertTo(img_noised, CV_8U);

		// denoising with median filtering
		if(median_filter_size%2)
			cv::medianBlur(img_noised, img_denoised, median_filter_size);

		cv::imshow(win_name, img);
		cv::imshow("With noise", img_noised);
		cv::imshow("Denoised with median filtering", img_denoised);

		// denoising with bilateral filtering
		cv::bilateralFilter(img_noised, img_denoised, 0, bilateral_sigma_r_x10/10.0, bilateral_sigma_s_x10/10.0f);
		cv::imshow("Denoised with bilateral filtering", img_denoised);
	}
}

int main() 
{
	try
	{
		img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		
		img_float = img.clone();
		img_float.convertTo(img_float, CV_32F);
		img_gaussian_noise = img.clone();
		img_gaussian_noise.convertTo(img_gaussian_noise, CV_32F);
		img_denoised = img.clone();

		win_name = "Salt-and-pepper denoising";
		cv::namedWindow(win_name);
		cv::createTrackbar("gaussian-noise", win_name, &gaussian_noise_stdev_x10, 300, gaussianNoiseReduction);
		cv::createTrackbar("median-filter", win_name, &median_filter_size, 100, gaussianNoiseReduction);
		cv::createTrackbar("bilateral-sigma-r", win_name, &bilateral_sigma_r_x10, 300, gaussianNoiseReduction);
		cv::createTrackbar("bilateral-sigma-s", win_name, &bilateral_sigma_s_x10, 300, gaussianNoiseReduction);

		gaussianNoiseReduction(0, 0);
		cv::waitKey(0);

		return 1;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

