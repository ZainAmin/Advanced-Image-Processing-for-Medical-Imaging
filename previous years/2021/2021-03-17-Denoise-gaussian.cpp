// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/photo.hpp>

// converts sigma to the corresponding nonlocal means parameters
// as suggested by authors in their original paper
void NlMeansParameters(int sigma, float &h, int &N, int &S);

namespace
{
	cv::Mat input_img;
	std::string win_name;
	int sigma_gaussian_x10 = 1;

	// bilateral filter parameters
	int sigma_color_x10   = 1;
	int sigma_spatial_x10 = 1;

	// non-local means parameters
	int switch_nlm = 0;
	int signa_nlm = 1;

	void denoiseGaussian(int pos, void* userdata)
	{
		cv::Mat img_noised = input_img.clone();

		// skip denoising if there is no noise
		if(sigma_gaussian_x10 <= 0)
			return;

		// generate gaussian noise
		cv::Mat gaussian_noise(input_img.rows, input_img.cols, CV_32F);
		cv::randn(gaussian_noise, 0, sigma_gaussian_x10/10.0);

		// add gaussian on top of original image
		img_noised.convertTo(img_noised, CV_32F);
		img_noised += gaussian_noise;
		img_noised.convertTo(img_noised, CV_8U);

		cv::imshow(win_name, img_noised);


		// TODO: skip incorrect parameter configurations


		/*cv::Mat img_denoised_bilateral;
		cv::bilateralFilter(img_noised, img_denoised_bilateral, -1, sigma_color_x10/10.0, sigma_spatial_x10/10.0);
		cv::imshow("Bilateral-denoised", img_denoised_bilateral);
		cv::absdiff(img_noised, img_denoised_bilateral, img_denoised_bilateral);
		cv::normalize(img_denoised_bilateral, img_denoised_bilateral, 0, 255, cv::NORM_MINMAX);
		cv::imshow("Bilateral-difference", img_denoised_bilateral);*/

		if(switch_nlm == 0)
			return;

	
		cv::Mat img_denoised_nlm;
		float h;
		int N, S;
		NlMeansParameters(signa_nlm, h, N, S);
		cv::fastNlMeansDenoising(img_noised, img_denoised_nlm, h, N, S);
		cv::imshow("NLM-denoised", img_denoised_nlm);
		cv::absdiff(img_noised, img_denoised_nlm, img_denoised_nlm);
		cv::normalize(img_denoised_nlm, img_denoised_nlm, 0, 255, cv::NORM_MINMAX);
		cv::imshow("NLM-difference", img_denoised_nlm);
	}
}

// GOAL: reduce salt-and-pepper noise
int main() 
{
	try
	{
		// load the image
		input_img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
		if(!input_img.data)
			throw ucas::Error("Cannot read image");

		// create the UI
		win_name = "Denoising gaussian";
		cv::namedWindow(win_name);
		cv::createTrackbar("noise", win_name, &sigma_gaussian_x10, 200, denoiseGaussian);
		cv::createTrackbar("BF_sigma_color", win_name, &sigma_color_x10, 200, denoiseGaussian);
		cv::createTrackbar("BF_sigma_spatial", win_name, &sigma_spatial_x10, 200, denoiseGaussian);
		cv::createTrackbar("NLM_on", win_name, &switch_nlm, 1, denoiseGaussian);
		cv::createTrackbar("NLM_sigma", win_name, &signa_nlm, 100, denoiseGaussian);

		// start the UI
		denoiseGaussian(0, 0);

		// waits for user to press a button and exit from the app
		cv::waitKey(0);

		return EXIT_SUCCESS;
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


void NlMeansParameters(int sigma, float &h, int &N, int &S)
{
	h = 3.0f;		// OpenCV default value
	N = 7;			// OpenCV default value
	S = 21;			// OpenCV default value
	if(sigma > 0  && sigma <= 15)
	{
		h = 0.40f * sigma;
		N = 3;
		S = 21;
	}
	if(sigma > 15 && sigma <= 30 )
	{
		h = 0.40f * sigma;
		N = 5;
		S = 21;
	}
	if(sigma > 30 && sigma <= 45 )
	{
		h = 0.35f * sigma;
		N = 7;
		S = 35;
	}
	if(sigma > 45 && sigma <= 75 )
	{
		h = 0.35f * sigma;
		N = 9;
		S = 35;
	}
	if(sigma > 75 && sigma <= 100 )
	{
		h = 0.30f * sigma;
		N = 11;
		S = 35;
	}
}