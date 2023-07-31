// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"

// to use fast non local means
#include <opencv2/photo/photo.hpp>

namespace eiid
{
	// since we work with a GUI, one possible solution to pass parameters
	// to/from the GUI/application core functions is to store parameters 
	// (including images) in global variables
	cv::Mat img;

	bool bilateral = true;			// whether we are applying bilateral (true) or nonlocal means (false) denoising
	int gaussian_noise_sigma = 0;	// standard deviation of gaussian noise to add

	// bilateral filter parameters
	int filter_size = 7;
	int sigma_color = 0;
	int sigma_space = 0;

	// nonlocal means parameters
	int sigma_nlmeans = 0;

	// converts sigma to the corresponding nonlocal means parameters
	// as suggested by authors in their original paper
	void NlMeansParameters(int sigma, float& h, int& N, int& S)
	{
		h = 3.0f;		// OpenCV default value
		N = 7;			// OpenCV default value
		S = 21;			// OpenCV default value
		if (sigma > 0 && sigma <= 15)
		{
			h = 0.40f * sigma;
			N = 3;
			S = 21;
		}
		if (sigma > 15 && sigma <= 30)
		{
			h = 0.40f * sigma;
			N = 5;
			S = 21;
		}
		if (sigma > 30 && sigma <= 45)
		{
			h = 0.35f * sigma;
			N = 7;
			S = 35;
		}
		if (sigma > 45 && sigma <= 75)
		{
			h = 0.35f * sigma;
			N = 9;
			S = 35;
		}
		if (sigma > 75 && sigma <= 100)
		{
			h = 0.30f * sigma;
			N = 11;
			S = 35;
		}
	}

	// NOTE: this is a callback function we will link to the trackbars in the GUI
	//       all trackbar callback functions must have the prototype (int, void*)
	//       see http://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=createtrackbar
	void denoiseGaussian(int pos, void* userdata)
	{
		// convert to gray
		if (img.channels() != 1)
		{
			printf("Multichannel images not supported\n");
			return;
		}

		// add gaussian noise
		// NOTE: since gaussian noise adds random positive and negative fluctuations
		// around the mean 0, we need to make this operation in CV_32F and then
		// convert back to CV_8U
		cv::Mat img_noised = eiid::img.clone();
		if (gaussian_noise_sigma > 0)
		{
			cv::Mat gaussian_noise(eiid::img.rows, eiid::img.cols, CV_32F);
			cv::randn(gaussian_noise, cv::Scalar(0), cv::Scalar(eiid::gaussian_noise_sigma));
			img_noised.convertTo(img_noised, CV_32F);
			img_noised += gaussian_noise;
			img_noised.convertTo(img_noised, CV_8U);
		}

		// initialize denoised image
		cv::Mat img_denoised = img.clone();

		// perform denoising with either bilateral or nonlocal means filtering
		if (eiid::bilateral)
		{
			if (filter_size > 0)
				cv::bilateralFilter(img_noised, img_denoised, filter_size, sigma_color, sigma_space);
			else
				printf("Cannot apply bilateral filtering: filter size (%d) should be > 0\n", filter_size);
		}
		else
		{
			if (sigma_nlmeans > 0)
			{
				float h = 0;	// denoising strength
				int N = 0;		// patch size
				int S = 0;		// search window size
				NlMeansParameters(sigma_nlmeans, h, N, S);
				cv::fastNlMeansDenoising(img_noised, img_denoised, h, N, S);
			}
			else
				printf("Cannot apply nonlocal means filtering: sigma (%d) should be > 0\n", sigma_nlmeans);
		}

		cv::imshow("denoising", img_noised);
		cv::imshow("denoised", img_denoised);
		cv::Mat difference = cv::abs(img_denoised - img_noised);
		cv::normalize(difference, difference, 0, 255, cv::NORM_MINMAX);
		cv::imshow("difference", difference);
	}
}


int main()
{
	try
	{
		// load the image
		std::string img_name = "lena.png";
		eiid::img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/" + img_name, cv::IMREAD_UNCHANGED);
		if (!eiid::img.data)
			throw ucas::Error("cannot load image");

		// enable / disable bilateral
		eiid::bilateral = false;

		// convert to gray
		cv::cvtColor(eiid::img, eiid::img, cv::COLOR_BGR2GRAY);

		// create the window and insert the trackbar
		cv::namedWindow("denoising");
		cv::createTrackbar("gaussian_noise", "denoising", &eiid::gaussian_noise_sigma, 50, eiid::denoiseGaussian);

		if (eiid::bilateral == true)
		{
			cv::createTrackbar("filter_size", "denoising", &eiid::filter_size, 50, eiid::denoiseGaussian);
			cv::createTrackbar("s_color", "denoising", &eiid::sigma_color, 200, eiid::denoiseGaussian);
			cv::createTrackbar("s_space", "denoising", &eiid::sigma_space, 100, eiid::denoiseGaussian);
		}
		else
			cv::createTrackbar("s_nlmeans", "denoising", &eiid::sigma_nlmeans, 100, eiid::denoiseGaussian);

		// call function with default parameters so it is updated right after the app starts
		eiid::denoiseGaussian(0, 0);

		// wait for key press = windows stay opened until the user presses any key
		cv::waitKey(0);

		return EXIT_SUCCESS;
	}
	catch (aia::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

